import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, data_dict, sample_length):
        self.dict = data_dict
        self.sample_length = sample_length

    def __len__(self):
        return len(self.dict["features"]) + 1 - self.sample_length

    def __getitem__(self, idx):
        # s'il ne reste plus assez d'elements pour former un echantillon complet
        assert (len(self.img) - idx) >= self.sample_length, (
            f"probleme de taille, taille demo: {len(self.img)}, self.sample_length: {self.sample_length}, idx:{idx}, len(self.img)-(idx+1)): {len(self.img) - (idx + 1)} "
        )

        data = self.img[idx : idx + self.sample_length]

        return data


class CustomDataset(Dataset):
    def __init__(self, data_dict, sample_length):
        """
        Args:
            data_dict (dict): Dictionary containing dataset keys and lists of data.
            sample_length (int): Number of elements to return per sample.
        """
        self.data_dict = data_dict
        self.sample_length = sample_length

        # Extract episode indices to split episodes
        self.episode_indices = data_dict["episode_index"]
        self.episodes = self._split_by_episode()

    def _split_by_episode(self):
        """Splits the dataset into episodes based on the 'episode_index'."""
        episodes = []
        current_episode = []
        last_episode_idx = self.episode_indices[0]

        for i in range(len(self.episode_indices)):
            if self.episode_indices[i] != last_episode_idx:
                episodes.append(current_episode)
                current_episode = []
                last_episode_idx = self.episode_indices[i]

            current_episode.append(i)

        if current_episode:
            episodes.append(current_episode)

        return episodes

    def __len__(self):
        """Returns the total number of possible sequences across episodes."""
        return sum(len(ep) for ep in self.episodes)

    def __getitem__(self, idx):
        """Returns a sample of length 'sample_length', ensuring no episode mixing."""
        # Find which episode the index belongs to
        episode = None
        cumulative_idx = 0

        for ep in self.episodes:
            if idx < cumulative_idx + len(ep):
                episode = ep
                break
            cumulative_idx += len(ep)

        if episode is None:
            raise IndexError("Index out of dataset range.")

        # Get start index within the episode
        local_idx = idx - cumulative_idx
        end_idx = local_idx + self.sample_length

        # Ensure we do not go beyond the episode length
        if end_idx > len(episode):
            end_idx = len(episode)

        indices = episode[local_idx:end_idx]

        # Repeat the last frame if not enough data
        while len(indices) < self.sample_length:
            indices.append(indices[-1])

        # Create the sample dictionary
        if self.sample_length > 1:
            sample = {}
            for key in self.data_dict:
                stacked = torch.stack(
                    [self.data_dict[key][i] for i in indices]
                ).squeeze()
                if key in ["features", "action"]:
                    # flatten the features and actions
                    sample[key] = stacked.flatten()
                else:
                    sample[key] = stacked
        else:
            sample = {
                key: torch.stack([self.data_dict[key][i] for i in indices]).squeeze()
                for key in self.data_dict
            }

        return sample
