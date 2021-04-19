import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import sys
import glob

plt.rcParams.update({'font.size': 20})

colors_map = {
    'Meta-Train from Scratch': '#fca503',
    'Test': '#0040ff'
}

def read_file(vec, file_name, term):
    print(file_name)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return vec

        mean_reward = False
        for idx, line in enumerate(lines):
            if term not in line:
                continue
            epoch_idx = idx
            epoch_line = line
            while 'Epoch' not in epoch_line:
                epoch_idx -= 1
                epoch_line = lines[epoch_idx]

            epoch = int(epoch_line.split(' ')[1].split('\t')[0])

            floats = line.split('\t')[0]
            left_bracket = floats.find('[')
            right_bracket = floats.find(']')

            if left_bracket == -1 and left_bracket == -1:

                floats = line.split('\t')[0]
                if epoch > len(vec):
                    vec.append([float(floats.split(' ')[-1].strip())])
                else:
                    vec[epoch - 1].append(float(floats.split(' ')[-1].strip()))

            else:
                floats = np.fromstring(floats[left_bracket + 1:right_bracket], dtype=float, sep=' ')

                if epoch > len(vec):
                    vec.append([floats.mean()])
                else:
                    vec[epoch - 1].append(floats.mean())

    return vec

def parse_plot(files, term='Reward'):
    coll = dict()
    episode_coll = dict()
    for fname in files:
        f = fname.split('.')
        if 'train' in fname:
            label = 'Meta-Train from Scratch'
        elif 'test' in fname:
            label = 'Test'

        if label not in coll:
            coll[label] = []
            episode_coll[label] = []

        coll[label] = read_file(coll[label], fname, term)
        episode_coll[label] = read_file(episode_coll[label], fname, 'Episode')

    for label in coll.keys():
        coll[label] = coll[label][:250]
        episode_coll[label] = episode_coll[label][:250]

        mean_values = []
        max_values = []
        min_values = []

        for val in coll[label]:
            mean = sum(val) / len(val)

            if term == 'Success':
                mean *= 100
            mean_values.append(mean)
            variance = np.std(val)/(np.sqrt(len(val)))

            if term == 'Success':
                variance *= 100
            variance = variance if variance < 20 else 20
            max_values.append(mean + variance)
            min_values.append(mean - variance)

        mean_episodes = []
        for epi_val in episode_coll[label]:
            mean_episodes.append(sum(epi_val) / len(epi_val))

        print(label)
        print('max: ', np.max(mean_values))
        print('min: ', np.min(mean_values))
        max_idx = np.argmax(mean_values)
        min_idx = np.argmin(mean_values)
        print('max std: ', np.std(coll[label][max_idx]))
        print('min std: ', np.std(coll[label][min_idx]))

        plt.plot(np.arange(len(coll[label])), mean_values, linewidth=2.0, label=label, color=colors_map[label])
        plt.fill_between(np.arange(len(coll[label])), min_values, max_values, color=colors.to_rgba(colors_map[label], alpha=0.2))

        # plt.plot(mean_episodes, mean_values, linewidth=1.5, label=label, color=colors_map[label])
        # plt.fill_between(mean_episodes, min_values, max_values, color=colors.to_rgba(colors_map[label], alpha=0.2))

    plt.xlabel('Epochs')
    if term == 'Success':
        term = 'Success Rate (%)'
    plt.ylabel(term)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.legend(framealpha=1)
    plt.grid()
#     plt.title('GFootball {} {}'.format(sys.argv[2], term))

files = glob.glob(sys.argv[1] + "*")
files = list(filter(lambda x: x.find(".pt") == -1, files))

# 'Epoch'/ 'Steps-taken'
term = sys.argv[3]
parse_plot(files, term)
plt.show()