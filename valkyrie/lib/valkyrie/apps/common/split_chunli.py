import os

def split_file(input_file, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    current_file = None

    with open(input_file, 'r') as file:
        for line in file:
            # Check if the line is a separator
            if line.startswith('==> ') and line.endswith(' <==\n'):
                # Close the current file if it's open
                if current_file is not None:
                    current_file.close()

                # Extract the file name from the separator line
                file_path = line.replace('/',' ').split()[5:-1]# Adjust indices if necessary

                file_path = os.path.join(dest_dir, '/'.join(file_path))

                # Create directories if they do not exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Open a new file
                current_file = open(file_path, 'w')
            elif current_file is not None:
                # Write to the current file if it's open
                current_file.write(line)

    if current_file is None:
        raise Exception('No separator found in src file')

    # Close the last file
    if current_file is not None:
        current_file.close()


if '__main__' == __name__:
    #input_file = '/home/bb/src/chunli/chunli/fitter.csv'
    #input_file  = '/home/bb/src/chunli/chunli/post_trade.csv'
    input_file  = '/media/veracrypt1/chunli_20240214/fitter.csv'

    destination_directory = '/media/veracrypt1/chunli_20240214/fitter/fitter.csv'
    split_file(input_file, destination_directory)
