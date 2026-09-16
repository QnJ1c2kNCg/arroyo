use super::expiring_time_key_map::ExpiringTimeKeyTable;
use super::global_keyed_map::GlobalKeyedTable;
use super::{CheckpointParquetMetadata, Table, TableEpochCheckpointer};
use crate::{CheckpointMessage, TableData};
use arrow::compute::{concat_batches, filter_record_batch, take};
use arrow_array::cast::AsArray;
use arrow_array::{
    ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray, TimestampNanosecondArray,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use arroyo_rpc::df::ArroyoSchema;
use arroyo_rpc::get_hasher;
use arroyo_rpc::grpc::rpc::{
    ExpiringKeyedTimeTableCheckpointMetadata, ExpiringKeyedTimeTableConfig, GlobalKeyedTableConfig,
    GlobalKeyedTableTaskCheckpointMetadata,
};
use arroyo_storage::StorageProvider;
use arroyo_types::{TaskInfo, range_for_server, server_for_hash};
use datafusion::common::hash_utils::create_hashes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::mpsc;

fn checkpoint() -> CheckpointMessage {
    CheckpointMessage {
        epoch: 1,
        time: SystemTime::UNIX_EPOCH,
        watermark: None,
        then_stop: false,
    }
}

fn task_info(partition: usize, parallelism: usize) -> Arc<TaskInfo> {
    let mut task = TaskInfo::for_test("checkpoint-test", "operator");
    task.task_index = partition as u32;
    task.parallelism = parallelism as u32;
    task.key_range = range_for_server(partition, parallelism);
    Arc::new(task)
}

fn hashes(batch: &RecordBatch) -> Vec<u64> {
    let keys = batch.project(&[0, 1]).unwrap();
    let mut hashes = vec![0; batch.num_rows()];
    create_hashes(keys.columns(), &get_hasher(), &mut hashes).unwrap();
    hashes
}

fn partition(batch: &RecordBatch, partition: usize, parallelism: usize) -> RecordBatch {
    let mask = BooleanArray::from(
        hashes(batch)
            .into_iter()
            .map(|hash| server_for_hash(hash, parallelism) == partition)
            .collect::<Vec<_>>(),
    );
    filter_record_batch(batch, &mask).unwrap()
}

fn sort_by_row_id(batch: RecordBatch) -> RecordBatch {
    let indices = sort_to_indices(batch.column(2), None, None).unwrap();
    RecordBatch::try_new(
        batch.schema(),
        batch
            .columns()
            .iter()
            .map(|column| take(column, &indices, None).unwrap())
            .collect(),
    )
    .unwrap()
}

#[tokio::test]
async fn keyed_checkpoint_restarts_and_rescales_with_stable_hashes() {
    let temp = tempfile::tempdir().unwrap();
    let url = temp.path().to_str().unwrap();
    let storage = Arc::new(StorageProvider::for_url(url).await.unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("region", DataType::Utf8, true),
        Field::new("row_id", DataType::Int64, false),
        Field::new(
            "_timestamp",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            (0..96)
                .map(|i| (i % 5 != 0).then_some(i % 13))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            (0..96)
                .map(|i| match i % 3 {
                    0 => None,
                    1 => Some("east"),
                    _ => Some("west"),
                })
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from_iter_values(0..96)),
        Arc::new(TimestampNanosecondArray::from(vec![1_000_000_000; 96])),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let config = ExpiringKeyedTimeTableConfig {
        table_name: "keyed".into(),
        description: "checkpoint recovery test".into(),
        retention_micros: 60_000_000,
        schema: Some(ArroyoSchema::new_keyed(schema.clone(), 3, vec![0, 1]).into()),
        generational: false,
    };

    let mut subtasks = HashMap::new();
    for subtask in 0..2 {
        let rows = partition(&batch, subtask, 2);
        assert!(rows.num_rows() > 0);
        let table = ExpiringTimeKeyTable::from_config(
            config.clone(),
            task_info(subtask, 2),
            storage.clone(),
            None,
            0,
        )
        .unwrap();
        let mut writer = table.epoch_checkpointer(1, None).unwrap();
        // Different batch boundaries must not change the hashes persisted on disk.
        let midpoint = rows.num_rows() / 2;
        writer
            .insert_data(TableData::RecordBatch(rows.slice(0, midpoint)))
            .await
            .unwrap();
        writer
            .insert_data(TableData::RecordBatch(
                rows.slice(midpoint, rows.num_rows() - midpoint),
            ))
            .await
            .unwrap();
        let (metadata, bytes) = writer.finish(&checkpoint()).await.unwrap().unwrap();
        assert!(bytes > 0);
        let file = &metadata.files[0];
        let reader = ParquetRecordBatchReaderBuilder::try_new(
            storage.get(file.file.as_str()).await.unwrap(),
        )
        .unwrap();
        for stored in reader.build().unwrap() {
            let stored = stored.unwrap();
            assert_eq!(
                stored
                    .column(4)
                    .as_primitive::<arrow_array::types::UInt64Type>()
                    .values()
                    .as_ref(),
                hashes(&stored)
            );
        }
        let expected_hashes = hashes(&rows);
        assert_eq!(file.min_routing_key, *expected_hashes.iter().min().unwrap());
        assert_eq!(file.max_routing_key, *expected_hashes.iter().max().unwrap());
        subtasks.insert(subtask as u32, metadata);
    }
    let metadata = ExpiringTimeKeyTable::merge_checkpoint_metadata(config.clone(), subtasks)
        .unwrap()
        .unwrap();
    storage
        .put("checkpoint.pb", metadata.encode_to_vec())
        .await
        .unwrap();
    drop(storage);

    // Reopen real files with new providers/tables: same parallelism, scale up, scale down.
    for parallelism in [2, 5, 1] {
        let storage = Arc::new(StorageProvider::for_url(url).await.unwrap());
        let metadata = ExpiringKeyedTimeTableCheckpointMetadata::decode(
            storage.get("checkpoint.pb").await.unwrap(),
        )
        .unwrap();
        let mut restored_count = 0;
        for subtask in 0..parallelism {
            let table = ExpiringTimeKeyTable::from_config(
                config.clone(),
                task_info(subtask, parallelism),
                storage.clone(),
                Some(metadata.clone()),
                0,
            )
            .unwrap();
            let (tx, _rx) = mpsc::channel(1);
            let view = table.get_view(tx, None).await.unwrap();
            let batches: Vec<_> = view
                .all_batches_for_watermark(None)
                .flat_map(|(_, batches)| batches.iter())
                .collect();
            let restored = concat_batches(&schema, batches).unwrap();
            restored_count += restored.num_rows();
            assert_eq!(
                sort_by_row_id(restored),
                sort_by_row_id(partition(&batch, subtask, parallelism))
            );
        }
        assert_eq!(restored_count, batch.num_rows());
    }
}

#[tokio::test]
async fn global_checkpoint_preserves_values_and_parquet_version_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let url = temp.path().to_str().unwrap();
    let storage = Arc::new(StorageProvider::for_url(url).await.unwrap());
    let config = GlobalKeyedTableConfig {
        table_name: "global".into(),
        uses_two_phase_commit: false,
        ..Default::default()
    };
    let table =
        GlobalKeyedTable::from_config(config.clone(), task_info(0, 1), storage.clone(), None, 7)
            .unwrap();
    let mut writer = table.epoch_checkpointer(1, None).unwrap();
    writer
        .insert_data(TableData::KeyedData {
            key: bincode::encode_to_vec(42u64, crate::BINCODE_CONFIG).unwrap(),
            value: bincode::encode_to_vec("checkpoint value".to_owned(), crate::BINCODE_CONFIG)
                .unwrap(),
        })
        .await
        .unwrap();
    let (metadata, bytes) = writer.finish(&checkpoint()).await.unwrap().unwrap();
    assert!(bytes > 0);
    let contents = storage
        .get(metadata.file.as_deref().unwrap())
        .await
        .unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(contents).unwrap();
    assert_eq!(
        CheckpointParquetMetadata::from(reader.metadata().file_metadata().key_value_metadata())
            .state_version,
        7
    );
    let metadata =
        GlobalKeyedTable::merge_checkpoint_metadata(config.clone(), HashMap::from([(0, metadata)]))
            .unwrap()
            .unwrap();
    storage
        .put("checkpoint.pb", metadata.encode_to_vec())
        .await
        .unwrap();
    drop(table);
    drop(storage);

    let storage = Arc::new(StorageProvider::for_url(url).await.unwrap());
    let metadata =
        GlobalKeyedTableTaskCheckpointMetadata::decode(storage.get("checkpoint.pb").await.unwrap())
            .unwrap();
    let restored =
        GlobalKeyedTable::from_config(config, task_info(0, 1), storage, Some(metadata), 7).unwrap();
    let (tx, _rx) = mpsc::channel(1);
    let view = restored.memory_view::<u64, String>(tx).await.unwrap();
    assert_eq!(
        view.get_all(),
        &HashMap::from([(42, "checkpoint value".to_owned())])
    );
}
