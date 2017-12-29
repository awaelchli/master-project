import GTAV2, VIPER


test_set = VIPER.Subsequence(
    folder=VIPER.FOLDERS['val'],
    sequence_length=100,
    overlap=0,
    transform=None,
    max_size=0,
    relative_pose=False
)

train_set = VIPER.Subsequence(
    folder=VIPER.FOLDERS['train'],
    sequence_length=100,
    overlap=0,
    transform=None,
    max_size=0,
    relative_pose=False
)

test_set2 = GTAV2.Subsequence(
    folder=GTAV2.FOLDERS['test'],
    sequence_length=100,
    overlap=0,
    transform=None,
    max_size=0,
    relative_pose=False
)

train_set2 = GTAV2.Subsequence(
    folder=GTAV2.FOLDERS['train'],
    sequence_length=100,
    overlap=0,
    transform=None,
    max_size=0,
    relative_pose=False
)

print('VIPER')
print('Statistics for test set:')
test_set.print_statistics()

print('')
print('Statistics for training set:')
train_set.print_statistics()

print('GTA')
print('Statistics for test set:')
test_set2.print_statistics()

print('')
print('Statistics for training set:')
train_set2.print_statistics()





