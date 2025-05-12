import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from nltk.tokenize import word_tokenize

class ImageCaptionDataset(Dataset):
    def __init__(self, captions_df, images_folder, transform, word_to_idx, max_length=50):
        self.captions_df = captions_df
        self.images_folder = images_folder
        self.transform = transform
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        # Obtener la imagen y su caption
        img_name = self.captions_df.iloc[idx]['image']
        caption = self.captions_df.iloc[idx]['caption']

        # Cargar y transformar la imagen
        image = Image.open(os.path.join(self.images_folder, img_name)).convert('RGB')
        image = self.transform(image)

        # Tokenizar y convertir caption a índices
        tokens = word_tokenize(caption.lower().strip())
        indices = [self.word_to_idx['<START>']]
        indices.extend([self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in tokens])
        indices.append(self.word_to_idx['<END>'])

        # Padding para que todas las secuencias tengan la misma longitud
        if len(indices) < self.max_length:
            indices.extend([self.word_to_idx['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length-1] + [self.word_to_idx['<END>']]

        return image, torch.tensor(indices)