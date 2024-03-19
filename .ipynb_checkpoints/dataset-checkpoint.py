#dataset for work
class BirdDataset(Dataset):
    def __init__(self, data, sr=32000, n_mels=128, fmin=0, fmax=None, duration=5, 
                 step=None, res_type="kaiser_fast", resample=True, valid=False, transform=None):
        self.data = data
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2
        
        self.transform = transform

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.valid = valid
        self.path = '' if valid else '/kaggle/input/birdclef-2023/train_audio/'
        self.res_type = res_type
        self.resample = resample

    def __len__(self):
        return len(self.data)
    
    def normalize(self, image):
        image = image.astype("float32", copy=False) / 255.0
        if image.shape[1] > 256:
            image = image[:128, :256]
        else:
          zeroes = np.zeros((128, 256 - image.shape[1]))
          image = np.concatenate([image, zeroes], axis=1, dtype=np.float32)
          
        image = np.stack([image, image, image], axis=0)
        return image
    
    def audio_to_image(self, audio):
        melspec = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax) 
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, row):
        filepath = self.path + str(row['path'])
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
          
        if self.valid:
            audios = []
            for i in range(self.audio_length, len(audio) + self.step, self.step):
                start = max(0, i - self.audio_length)
                end = start + self.audio_length
                audios.append(audio[start:end])

            if len(audios[-1]) < self.audio_length:
                audios = audios[:-1]

            images = [self.audio_to_image(audio) for audio in audios]
            images = np.stack(images)
            
        else:
            images = self.audio_to_image(audio)  
        
        labels = torch.tensor(row[3:]).float() 
        return (images, labels)
    
        
    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx])