import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ligandmpnn.get_params <output_dir>")
        sys.exit(1)
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    files = [
        # Original ProteinMPNN weights
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt",
            "proteinmpnn_v_48_002.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt",
            "proteinmpnn_v_48_010.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt",
            "proteinmpnn_v_48_020.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt",
            "proteinmpnn_v_48_030.pt",
        ),
        # LigandMPNN with num_edges=32; atom_context_num=25
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt",
            "ligandmpnn_v_32_005_25.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt",
            "ligandmpnn_v_32_010_25.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt",
            "ligandmpnn_v_32_020_25.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt",
            "ligandmpnn_v_32_030_25.pt",
        ),
        # Per residue label membrane ProteinMPNN
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/per_residue_label_membrane_mpnn_v_48_020.pt",
            "per_residue_label_membrane_mpnn_v_48_020.pt",
        ),
        # Global label membrane ProteinMPNN
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/global_label_membrane_mpnn_v_48_020.pt",
            "global_label_membrane_mpnn_v_48_020.pt",
        ),
        # SolubleMPNN
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt",
            "solublempnn_v_48_002.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt",
            "solublempnn_v_48_010.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
            "solublempnn_v_48_020.pt",
        ),
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt",
            "solublempnn_v_48_030.pt",
        ),
        # LigandMPNN for side-chain packing (multi-step denoising model)
        (
            "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt",
            "ligandmpnn_sc_v_32_002_16.pt",
        ),
    ]

    def download_file(url, dest):
        try:
            response = requests.get(url, stream=True, timeout=60)
            total = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as file, tqdm(
                desc=os.path.basename(dest),
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
                bar.set_description(f"Saved to {dest}")
            return 0
        except Exception as e:
            return f"Failed to download {url}: {e}"

    print(f"Starting download of {len(files)} files with multithreading...")
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(download_file, url, os.path.join(out_dir, fname)): fname
            for url, fname in files
        }
        for future in as_completed(future_to_file):
            fname = future_to_file[future]
            result = future.result()
            if result != 0:
                print(result)


if __name__ == "__main__":
    main()
