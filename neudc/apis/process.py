from neudc.core import select_instances

def process_video(model,
                  indexer,
                  dataset,
                  dataloader,
                  saver) -> None:

    # for each image in buffer
    # dataloader has no len
    for batch in dataloader:
        # call forward
        predictions = model(batch)
        target_bboxes, target_masks, filter_mask = select_instances(predictions)
        dataloader.update(filter_mask)
        saver.save()
