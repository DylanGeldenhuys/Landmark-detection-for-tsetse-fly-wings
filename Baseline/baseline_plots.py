import matplotlib.pyplot as plt
from baseline_utils import load_annotations, sanity_plot, combine_and_split, compute_baseline, mean_test_error

# Python file replicating notebook that generates baseline plots
# Configure annotations and root paths before running

# file path for annotations
left_annotations_file = '../../DATA/annotations_left.txt'
right_annotations_file = '../../DATA/annotations_right.txt'
root_path_left = '../../DATA/images_left/'
root_path_right = '../../DATA/images_right/'

left_img_names, left_coordinates, left_count = load_annotations(left_annotations_file)
right_img_names, right_coordinates, right_count = load_annotations(right_annotations_file, flip_right=True)

sanity_plot(root_path_left, left_img_names, left_coordinates)
sanity_plot(root_path_right, right_img_names, right_coordinates, right_img = True)


# Data split and baseline implementation
# The data is split and the baseline is implemented by calculating the mean coordinate for each landmark.
training_landmarks, test_landmarks = combine_and_split(left_coordinates, right_coordinates)
train_mean, train_std = compute_baseline(training_landmarks, plot_mean = False)
landmark_errors = mean_test_error(test_landmarks, train_mean)

# plot landmark errors
plt.figure(figsize=(10,10))
plt.boxplot(landmark_errors)
plt.xlabel('landmarks', fontsize=8)
plt.ylabel('pixel distance error', fontsize=8)
plt.savefig('baseline_boxplot.pdf', bbox_inches='tight')
plt.show()

