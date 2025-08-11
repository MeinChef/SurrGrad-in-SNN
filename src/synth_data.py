import torch

def generate_spikes(
    isi: torch.Tensor,
    inputs: int,
    r: int = 13,
    T: int = 1000,
    n_samples: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to generate Spikes, adapted from paper https://arxiv.org/abs/2507.16043v1
    isi has to be of shape [classes] and be an integer tensor that specifies the inter-spike-intervals for each class
    """
    
    no_spk_pairs = r * (T/1000)
    n_classes = len(isi)
    output = torch.zeros(
        size = (T, n_samples, inputs),
        dtype = torch.float32
    )

    samples_per_class_val = int(n_samples/ n_classes)
    samples_per_class = torch.full_like(
        input = isi,
        fill_value = samples_per_class_val,
        dtype = torch.int32
    )
    # make sure that we have n_samples in the array. Increase the last class with the remaining values
    # dirty solution, but works
    samples_per_class[-1] += n_samples - (n_classes * samples_per_class_val)


    for i, sample in enumerate(samples_per_class):
        time_stamps = torch.randint(
            low = 0,
            high = T - isi[i],
            size = (sample, no_spk_pairs)
        )
        time_stamps = torch.sort(time_stamps)

        bad_step_idx = []
        for j, step in enumerate(time_stamps):
            if j == 0: 
                continue
            if step - isi[i] <= time_stamps[j-1]:
                bad_step_idx.append(j)
            if step != time_stamps[-1]:
                if step + isi[i] > time_stamps[j+1] and time_stamps[j-1] not in bad_step_idx:

        for step in bad_step_idx:
            # generate new step
            # substract step from the whole array, take min of absolute value, that's the closest index i think
            # if great, replace in time_stamps, sort time_stamps and repeat
            pass
        # do all of that at once and in-place
        # [3,8,10,14]
        # [-3, -1]   


    target = torch.stack([torch.full(size, cls) for cls, size in enumerate(samples_per_class)])
