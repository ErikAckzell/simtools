import os

# Definitions for parsing.
code_marker = "#!! Include:"
code_marker_end = "#!! Include end"
code_files = [
            "BDF.py",
            "step.py",
        ]

# Where to put the *.tex files.
output_dir = os.path.join("TeX", "includes","code")


# Output name template.
output_filename_template = "{}.tex"


def extract_code(code_buffer, data):
    """ Extract the code snippets marked with code_marker and put them in a
    dict keyed to the task they were marked for. """


    def init_list(key):
        """ Make sure that there is a list for a given key. """
        if not code_buffer.get(key):
            code_buffer[key] = []


    def add_code(key, code):
        """ Add code to code_buffer, make sure there is a list first. """
        init_list(key)
        code_buffer[key].append(code)


    def parse_code_marker(line):
        """ Parse the code marker line and extract which tasks that should
        include the code, return a list of those tasks. """
        tasks_string = line.strip().replace(code_marker, '')
        return [task.strip() for task in tasks_string.split(',')]


    code_segment_buffer = []
    code_segment_for = ""

    for line in data:
        # Toggle inside or outside code snippet.
        if line.strip().startswith(code_marker):
            code_segment_for = parse_code_marker(line)
            continue

        # Not in a code snippet -> continue.
        if not code_segment_for:
            continue

        # In a code snippet and hit the end, add code to dict.
        if line.strip().startswith(code_marker_end):
            code = '\n'.join(code_segment_buffer)
            for task in code_segment_for:
                add_code(task, code)
            # Reset code_segment_for.
            code_segment_for = ""
            # Reset buffer.
            code_segment_buffer = []
        else: # Otherwise we are in a code segment and should append to buffer.
            # Avoid dubble newlines.
            code_segment_buffer.append(line.strip('\n'))


def parse_files():
    """ Parse all files in the code_files list and send the data to the code
    extractor together with the code_buffer dictionary. """

    def codelisting_wrap(code):
        """ Wrap the code in appropriate LaTeX syntax. """
        return "\\begin{{lstlisting}}\n{}\n\\end{{lstlisting}}".format(code)


    code_buffer = {}

    # Iterate over all the files.
    for filename in code_files:
        data = open(filename).readlines()
        extract_code(code_buffer, data)

    # Iterate over all tasks and their code data.
    for task, data_list in code_buffer.items():
        code_environment_list = []
        # Wrap each code section in it's own LaTeX environment.
        for data in data_list:
            code_environment_list.append(codelisting_wrap(data))
        output_filename = output_filename_template.format(task)
        output_path = os.path.join(output_dir, output_filename)
        # Write all the environment for the same task to the same file.
        with open(output_path, 'w') as handle:
            handle.write('\n'.join(code_environment_list))

def main():

    # Check if output dir exists.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Parse files.
    parse_files()

if __name__ == "__main__":
    main()
