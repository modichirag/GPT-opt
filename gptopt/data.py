from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

def load_data(name, batch_size):
    if name == 'tiny_shakespeare':
        return load_tiny_shakespeare(batch_size)
    elif name == 'ptb_text_only':
        return load_ptb_text_only(batch_size)
    elif name == 'wikitext-103':
        return load_wikitext('wikitext-103-v1', batch_size)  # Reuse load_wikitext for wikitext-103
    else:
        return load_wikitext(name, batch_size)

def load_wikitext(name, batch_size):
    """General function to load WikiText datasets."""
    dataset = load_dataset('wikitext', name, trust_remote_code=True)
    print(f"Number of training samples: {len(dataset['train'])}")
    print(f"Number of validation samples: {len(dataset['validation'])}")
    print(f"Number of test samples: {len(dataset['test'])}")
    print("First few lines of the dataset:")
    print(dataset['train'][:4]['text'])  # Print the first 4 lines
    # Create a DataLoader for the dataset
    train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset['test'], batch_size=batch_size)
    return train_dataloader, test_dataloader

def load_ptb_text_only(batch_size):
    dataset = load_dataset('ptb_text_only', trust_remote_code=True)
    dataset['train'] = dataset['train'].rename_column('sentence', 'text')
    dataset['test'] = dataset['test'].rename_column('sentence', 'text')
    print(f"Number of training samples: {len(dataset['train'])}")
    print(f"Number of validation samples: {len(dataset['validation'])}")
    print(f"Number of test samples: {len(dataset['test'])}")
    print("First few lines of the dataset:")
    print(dataset['train'][:4]['text'])  # Print the first 4 lines
    # Create a DataLoader for the dataset
    train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset['test'], batch_size=batch_size)
    return train_dataloader, test_dataloader

def load_tiny_shakespeare(batch_size):
    dataset = load_dataset('tiny_shakespeare', trust_remote_code=True)
    train = dataset['train'][0]['text']
    test = dataset['test'][0]['text']
    # Define chunk size (e.g., 100 characters)
    chunk_size = 500
    # Split the text into chunks of `chunk_size`
    trainchunks = [train[i:i + chunk_size] for i in range(0, len(train), chunk_size)]
    testchunks = [train[i:i + chunk_size] for i in range(0, len(test), chunk_size)]
    # Create a new dataset with these chunks
    train_dataset = Dataset.from_dict({'text': trainchunks})
    test_dataset = Dataset.from_dict({'text': testchunks})
    # Create a DataLoader using the new dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

