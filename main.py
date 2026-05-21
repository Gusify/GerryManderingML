import torch

def main():
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    x = torch.rand(3, 3, device=device)
    print("Sample tensor:\n", x)

if __name__ == "__main__":
    main()
