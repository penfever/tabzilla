import argparse
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Import all preprocessor modules and add them to list for them to be in list of preprocessors
import tabzilla_preprocessors_openml

preprocessor_modules = [tabzilla_preprocessors_openml]

dataset_path = Path("datasets")


def build_preprocessors_dict():
    preprocessors = {}
    for module in preprocessor_modules:
        duplicates = preprocessors.keys() & module.preprocessor_dict.keys()
        if duplicates:
            raise RuntimeError(
                f"Duplicate dataset_name key found in module {module}: {duplicates}"
            )
        preprocessors.update(module.preprocessor_dict)
    return preprocessors


preprocessors = build_preprocessors_dict()


def preprocess_dataset(dataset_name, overwrite=False, verbose=True):
    dest_path = dataset_path / dataset_name
    if not overwrite and dest_path.exists():
        if verbose:
            print(f"{dataset_name:<40}| Found existing folder. Skipping.")
        return dest_path

    print(f"{dataset_name:<40}| Processing...")
    if dataset_name not in preprocessors:
        raise KeyError(f"Unrecognized dataset name: {dataset_name}")
    dataset = preprocessors[dataset_name]()
    dataset.write(dest_path, overwrite=overwrite)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset pre-processing utility.")
    parser.add_argument("--dataset_name", help="Dataset to pre-process.")
    parser.add_argument(
        "--process_all",
        action="store_true",
        help="Use this flag to pre-process all datasets.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Use this flag to overwrite datasets."
    )
    parser.add_argument(
        "--print_dataset_names",
        action="store_true",
        help="Optional flag to print all dataset names. If this flag is set, no datasets will be pre-processed.",
    )
    args = parser.parse_args()

    if args.print_dataset_names:
        print("------------------------\n")
        print("Valid dataset names:\n")
        for i, dataset_name in enumerate(sorted(preprocessors.keys())):
            print(f"{i + 1}: {dataset_name} ")
        print("------------------------")
        sys.exit()

    if args.dataset_name is not None and args.process_all:
        raise RuntimeError(
            "dataset_name cannot be specified simultaneously with the flag process_all"
        )
    elif args.dataset_name is None and not args.process_all:
        raise RuntimeError("Need to specify either dataset_name or process_all flag")

    if args.process_all:
        for dataset_name in sorted(preprocessors.keys()):
            _ = preprocess_dataset(dataset_name, args.overwrite)
            print("Processed dataset {}".format(dataset_name))
    else:
        _ = preprocess_dataset(args.dataset_name, args.overwrite)
        print("Processed dataset {}".format(args.dataset_name))
