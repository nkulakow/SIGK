import os
print('Current directory:', os.getcwd())
if os.path.exists('data'):
    print('Data directory exists')
    contents = os.listdir('data')
    print('Contents:', contents)
    if 'DIV2K_train_256' in contents:
        train_files = os.listdir('data/DIV2K_train_256')
        print('Train files count:', len(train_files))
    else:
        print('DIV2K_train_256 not found')
else:
    print('Data directory does not exist')