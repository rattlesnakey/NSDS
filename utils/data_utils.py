from datasets import load_dataset


def load_calibration_data(
    dataset_name="monology/pile-uncopyrighted",
    num_samples=128,
    streaming=True,
):
    dataset = load_dataset(dataset_name, split="train", streaming=streaming)
    return list(dataset.take(num_samples))


def get_texts(data, field="text"):
    return [item[field] for item in data if item.get(field, "").strip()]
