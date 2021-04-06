

def LoadNumpyPickles(pickle_path, X_file, y_file, ID_file, DelNan = False):

    # Your pickle should be placed in prepData folder
    os.chdir(pickle_path)
    X = np.load(pickle_path + X_file, allow_pickle='TRUE')
    y = np.load(pickle_path + y_file, allow_pickle='TRUE')
    ID_frame = np.load(pickle_path + ID_file, allow_pickle='TRUE')


    return X, y, ID_frame






