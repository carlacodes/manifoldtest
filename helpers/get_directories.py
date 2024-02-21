import os


def get_home_dir():
    # determine operating system
    if os.name == 'nt':
        home_dir = 'D:/analysis'  # WINDOWS
    elif os.name == 'posix':  # Linux or Mac OS
        home_dir = "/media/jake/LaCie"  # Linux/Ubuntu
    return home_dir


def get_data_dir(animal, session):
    home_dir = get_home_dir()
    data_dir = os.path.join(home_dir, animal, session)
    return data_dir


def get_robot_maze_directory():
    if os.name == 'nt':
        home_dir = 'C:/Users/Jake/Documents/robot_code/robot_maze'  # WINDOWS C:\Users\Jake\Documents\robot_code\robot_maze
    elif os.name == 'posix':  # Linux or Mac OS
        home_dir = "/home/jake/Documents/robot_maze"  # Linux/Ubuntu
    return home_dir


def reverse_date(date):
    """
    Reverse the date from dd-mm-yyyy to yyyy-mm-dd.

    Parameters
    ----------
    date : str
        The date in dd-mm-yyyy format.

    Returns
    -------
    reversed_date : str
        The date in yyyy-mm-dd format.
    """
    date_split = date.split('-')
    reversed_date = '-'.join(date_split[::-1])
    return reversed_date


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    reversed_data = reverse_date(session)
    data_dir = get_data_dir(animal, session)
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    print(bonsai_dir)