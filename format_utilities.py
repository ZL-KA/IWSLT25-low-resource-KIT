import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", choices=["format-from-csv", "combine-hyps"])
    parser.add_argument("--file-paths", help="File paths seperated by |")
    parser.add_argument("--nr-samples", help="Nr samples seperated by |")
    parser.add_argument("--with-ref", default="False", choices=["True", "False"])
    parser.add_argument("--out-combined", help="Location to save the combined hyps file")

    args = parser.parse_args()

    if args.function == "format-from-csv":
        for path in args.file_paths.split('|'):
            assert '.csv' in path
            df = pd.read_csv(path, sep='|')
            hyps = df['pred_translation'].tolist()

            hyps_flatten = []
            n = None
            for hyp in hyps:
                if hyp == ' ':
                    hyps_flatten.extend([' '] * n)
                    continue

                hyps_flatten.extend(hyp.split('<SS>'))
                if n is None:
                    n = len(hyp.split('<SS>'))
                else:
                    assert n == len(hyp.split('<SS>'))

            if args.with_ref == "True":
                src = df['transcript'].tolist()
                ref = df['translation'].tolist()
                write_text_file(src, path.replace('.csv', '.src'))
                write_text_file(ref, path.replace('.csv', '.ref'))
            write_text_file(hyps_flatten, path.replace('.csv', '.hyps'))

    elif args.function == "combine-hyps":
        paths = args.file_paths.split('|')
        hyps = [load_text_file(path) for path in paths]
        nr_samples = args.nr_samples.split('|')
        nr_samples = [int(x) for x in nr_samples]
        combined = combine_lists(hyps, nr_samples)

        write_text_file(combined, args.out_combined)


def combine_lists(lists, n_values):
    assert len(lists) == len(n_values)
    splitted_lists = []
    nr_chunks = None
    for l, n in zip(lists, n_values):
        assert len(l) % n == 0
        chunked = [l[i:i + n] for i in range(0, len(l), n)]
        splitted_lists.append(chunked)
        if nr_chunks is None:
            nr_chunks = len(chunked)
        else:
            assert nr_chunks == len(chunked)
    combined = []
    for chunk_i in range(nr_chunks):
        for splitted_list in splitted_lists:
            combined.extend(splitted_list[chunk_i])
    return combined





def write_text_file(lines, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


def load_text_file(file_path):
    """
    Load text file into a list, each item is a line of the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


if __name__ == "__main__":
    main()
