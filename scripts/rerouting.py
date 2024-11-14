
def readlines_in_file(directory_path):
    # Initialize an empty string to store the concatenated content
    concatenated_content = []
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file has a '.txt' extension
        if filename.endswith('.graphml') and filename.startswith('2020') is False and filename.startswith('mask') is False:
            # Get the full file path
            file_path = os.path.join(directory_path, filename)
            # Open the text file and read its content
            with open(file_path, 'r', encoding='utf-8') as file:
                # concatenated_content += file.read() + "\n"  # Adding newline to separate files' content
                for line in file.readlines():
                    concatenated_content.append(line.strip())

    return concatenated_content



jeepney_routes = readlines_in_file('data/routes/relations.txt')


