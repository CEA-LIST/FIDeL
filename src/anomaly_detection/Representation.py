import torch
import torch.nn as nn
import torch.nn.functional as F
import ot


class Memory(nn.Module):
    def __init__(
        self,
        nb_of_episodes,
        frame_per_episode,
        patch_dim,
        features_dim,
        cfg,
    ):
        super(Memory, self).__init__()
        self.nb_of_episodes = nb_of_episodes
        self.frame_per_episode = frame_per_episode
        self.features_dim = features_dim
        self.patch_dim = patch_dim
        self.cfg = cfg
        self.flag_avg_has_been_computed = False

        # Use register_buffer so PyTorch handles device mapping and state_dict automatically
        self.register_buffer(
            "memory",
            torch.zeros(
                (frame_per_episode, self.patch_dim, features_dim), dtype=torch.float32
            ),
        )

        # Initialize a counter to compute mean and std
        self.register_buffer("counter", torch.zeros(frame_per_episode))

        print(
            f"frame_per_episode: {frame_per_episode}, patch_dim: {patch_dim}, features_dim: {features_dim}"
        )

        # Counter to compute sum((x-µ)**2), used only after µ is computed
        self.register_buffer(
            "memory_minus_mean",
            torch.zeros(
                (frame_per_episode, self.patch_dim, features_dim), dtype=torch.float32
            ),
        )

        self.memory_avg = 0
        self.memory_std = 0

    def update_memory(self, query, frame_idx):
        """
        query: tensor of shape (patch_size, feature_dim)
        frame_idx: feature's frame index
        """

        self.counter[frame_idx] += 1  # Keep track of summed features
        query = query.squeeze()

        # While µ (mean) is unknown
        if not self.flag_avg_has_been_computed:
            self.memory[frame_idx] += query

        # Only when µ is known
        if self.flag_avg_has_been_computed:
            self.memory_minus_mean[frame_idx] += (
                query - self.memory_avg[frame_idx]
            ) ** 2

    def compute_mean(self):
        # Compute mean along the episode dimension
        self.memory_avg = self.memory / self.counter.unsqueeze(1).unsqueeze(2)

    def compute_std(self):
        self.memory_std = torch.sqrt(
            self.memory_minus_mean / self.counter.unsqueeze(1).unsqueeze(2)
        )

    def get_mean_and_std(self):
        return self.memory_avg, self.memory_std

    def compute_ot_distance(self, query, keys_avg):
        """
        query: (nb_patches_q, features_dim)
        keys_avg: (frame_idx, nb_patches_k, features_dim)
        """

        Ms = []
        distances = []

        for t in range(keys_avg.shape[0]):
            key = keys_avg[t]  # (nb_patches_k, features_dim)

            # Euclidean cost matrix
            M = torch.cdist(query, key, p=2).cpu().detach().numpy()  # (P_q, P_k)

            # Uniform distributions (can be adjusted)
            a = ot.unif(query.shape[0])  # sum = 1
            b = ot.unif(key.shape[0])  # sum = 1

            # Solve OT with Earth Mover's Distance (Sinkhorn can be used for faster results)
            dist = ot.emd2(a, b, M)  # Minimum total cost

            distances.append(dist)
            Ms.append(M)

        distances = torch.tensor(distances)
        min_dist, min_idx = torch.min(distances, dim=0)
        M_best = Ms[min_idx].clone().detach()

        # Optional: project query patches to key
        # transport_plan = ot.emd(a, b, Ms[min_idx])
        # transported_features = transport_plan @ key

        dist_patch = M_best.min(axis=1, keepdims=True)[0]  # Minimum distance per patch

        return min_dist, min_idx, dist_patch

    def compute_distance(self, query):
        """
        query: tensor of shape (features_dim)
        Returns distance between key and query.
        """

        # Fetch feature's gaussian distributions over episode dimension
        keys_avg = self.memory_avg  # (frame_idx, patch_idx, avg_feature)
        keys_std = self.memory_std
        epsilon = 1e-5

        if self.cfg.anomaly_detection.distance_type == "optimal_transport":
            return self.compute_ot_distance(query, keys_avg)

        if self.cfg.anomaly_detection.distance_type == "euclidean_matrix":
            Ms = []
            distances = []

            for t in range(keys_avg.shape[0]):
                # Flatten keys for current frame
                key_mu = keys_avg[t]  # shape: (P, F)
                key_sigma = keys_std[t]  # shape: (P, F)
                q = query  # shape: (P, F)

                # Expand query and key tensors to enable pairwise comparison (P_q, P_k, F)
                q_exp = q.unsqueeze(1)  # (P_q, 1, F)
                mu_exp = key_mu.unsqueeze(0)  # (1, P_k, F)
                sigma_exp = key_sigma.unsqueeze(0)  # (1, P_k, F)

                # Compute Mahalanobis distance (without sqrt) between each query and key
                # D² = sum_f ( (q - µ)² / σ² )
                diff_sq = (q_exp - mu_exp) ** 2  # (P_q, P_k, F)
                inv_sigma_sq = 1.0 / (sigma_exp**2 + epsilon)  # (1, P_k, F)
                mahalanobis = diff_sq * inv_sigma_sq  # (P_q, P_k, F)

                # Sum over features to get scalar distance for each pair
                M = mahalanobis.sum(dim=-1)  # (P_q, P_k)

                # Compute total distance using minimum over columns (patch-to-key match)
                dist = M.min(dim=0, keepdim=True).values.sum()  # scalar

                # Store results
                distances.append(dist)
                Ms.append(M)

            distance = torch.tensor(distances)
            min_dist, min_idx = torch.min(distance, dim=0)
            M = Ms[min_idx].clone().detach()
            dist_patch = M.min(dim=1, keepdim=True).values

        # Compute distance per patch
        if self.cfg.anomaly_detection.distance_type == "euclidean":
            # Slight modification of the method, mixing Euclidean and OT
            if query.dim() == 1:
                query = query.unsqueeze(0)

            Ms = []
            distances = []
            for i in range(keys_avg.shape[0]):
                key = keys_avg[i].view(-1, keys_avg.shape[-1])
                if key.dim() == 1:
                    key = key.unsqueeze(0)

                # Euclidean cost matrix between each patch
                M = torch.cdist(query, key, p=2)
                dist = torch.sum(M.min(dim=1, keepdim=True).values)
                distances.append(dist)
                Ms.append(M)

            distance = torch.tensor(distances)
            min_dist, min_idx = torch.min(distance, dim=0)
            M = Ms[min_idx].clone().detach()
            dist_patch = M.min(dim=1, keepdim=True).values

        if self.cfg.anomaly_detection.distance_type == "cosine":
            # Normalize vectors to compute normalized scalar product
            average_keys_norm = F.normalize(
                keys_avg, p=2, dim=-1
            )  # (frame_index, patch_size, feature_size)
            query_norm = F.normalize(query, p=2, dim=-1)  # (patch_size, feature_size)

            # Compute cosine similarity (summing over the last dimension)
            cosine_sim = torch.einsum("ij,kij->k", query_norm, average_keys_norm)

            # Sum over the first dimension to get the distance per patch
            dist_patch = torch.abs(
                torch.einsum("ij,kij->i", query_norm, average_keys_norm)
            )
            distance = abs(1 - cosine_sim)  # Cosine distance

            # Minimum distance per frame
            min_dist, min_idx = torch.min(distance, dim=0)

        return min_dist, min_idx, dist_patch

    def forward(self, query, episode_idx, frame_idx, train):
        """
        query: tensor of shape (nb_patch, feature_dim)
        frame_idx: tensor of shape (int) indicating the time index of each demonstration
        """
        if train:
            # Update memory with new features
            self.update_memory(query, frame_idx)

            return 0, 0

        else:
            # In test mode, memory is not updated
            min_dist, min_idx, dist_patch = self.compute_distance(query)

            return min_dist, min_idx, dist_patch


class RepresentationNet(nn.Module):
    def __init__(
        self,
        nb_of_episodes,
        frame_per_episode,
        patch_dim,
        features_dim,
        cfg,
        train,
    ):
        super(RepresentationNet, self).__init__()
        self.frame_per_episode = frame_per_episode
        self.features_dim = features_dim
        self.patch_dim = patch_dim
        self.cfg = cfg
        self.train_mode = train
        self.proj_net = cfg.proj_net

        if self.proj_net:
            # Feature fusion block
            h_dim = self.features_dim  # Assuming one feature vector per frame
            self.bn = nn.Sequential(
                nn.BatchNorm1d(h_dim),
                nn.Linear(h_dim, h_dim // 2),
                nn.BatchNorm1d(h_dim // 2),
                nn.PReLU(),
                nn.Linear(h_dim // 2, h_dim // 4),
            )
            self.features_dim = h_dim // 4

        # Instantiate memory module with the new shape
        self.memory = Memory(
            nb_of_episodes,
            frame_per_episode,
            patch_dim,
            self.features_dim,
            cfg,
        )

    def forward(self, data):
        if self.train_mode:
            """
            data is a dictionary containing at least:
                - "features": extracted features tensor, shape (patch_size, feature_size)
                - "frame_index": indicates the frame number for a given episode
                - "episode_index": indicates the episode number = demonstration
                - potentially other info (action, timestamp, etc.)
            """
            # Extract features and frame index
            features = data[
                "features"
            ]  # Make sure to handle batch dimensions carefully instead of a raw squeeze()

            episode_index = int(data["episode_index"])
            frame_index = int(data["frame_index"])  # (batch,)

            features = features.to(self.cfg.device)
            features = features.float()

            # Remove batch dimension if batch size is 1
            if features.size(0) == 1:
                features = features.squeeze(0)

            if self.proj_net:
                # Pass through the fusion block
                features = self.bn(
                    features
                )  # Expected shape: patch_dim, features_dim // 4

            # Pass through the memory module
            memory_mean, memory_std = self.memory(
                features,
                episode_index,
                frame_index,
                train=self.train_mode,
            )
            return memory_mean, memory_std

        # Evaluation mode
        else:
            """
            data is a tensor:
                - "features": extracted features tensor, shape (patch_size, feature_size)
            """
            # Pass through the memory module
            if isinstance(data, dict):
                data = data["features"]
            data = data.to(self.cfg.device).float()

            min_dist, min_idx, dist_patch = self.memory(
                data,
                None,
                None,
                train=self.train_mode,
            )
            return min_dist, dist_patch


def representation(
    nb_of_episodes, frame_per_episode, patch_dim, features_dim, cfg, training
):
    representation_net = RepresentationNet(
        nb_of_episodes,
        frame_per_episode,
        patch_dim,
        features_dim,
        cfg,
        training,
    )

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = None
    if cfg.proj_net:
        optimizer = torch.optim.Adam(representation_net.parameters(), lr=1e-3)

    return representation_net, criterion, optimizer
