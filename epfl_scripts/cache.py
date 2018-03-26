"""
Implementation of cache to functions.

normal usage:

@cache_function("foo_{0}")
def foo(bar):
    #something

"""
import os
import pickle

savedFolder = "_cache_/"


def cache_function(name, forceLoad=False):
    """
    Decorator to mark functions as cached under a name.
    The name is formatted {.format()} with the function parameters, to allow parameters-dependent caches
    If the name is already cached, the value is returned without evaluating the function.
    If the name is not cached OR forceload=True, the function is evaluated, saved and returned

    Examples:   cache_function("mycache") will store the cache under 'mycache'
                cache_function("{0}_cache") will store foo(1) under '0_cache', foo('s') under 's_cache', and so on.
    """

    def func_decorator(func):
        def func_wrapper(*args, **kwargs):
            return cachedObject(name.format(*args, **kwargs), lambda: func(*args, **kwargs), forceLoad)

        return func_wrapper

    return func_decorator


###################### internal ############################


def saveObject(obj, name):
    """
    Save the object 'obj' under the name 'name'
    """
    checkFolder()
    with open(convertFilename(name), 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def loadObject(name):
    """
    Load the object saved under the name 'name'
    Returns the object or 'None' if not found
    """
    checkFolder()
    if not os.path.isfile(convertFilename(name)):
        return None
    with open(convertFilename(name), 'rb') as output:
        return pickle.load(output)


def cachedObject(name, default=lambda: None, forceLoad=False):
    """
    Implements a cached system.
    Tries to load the cached object with name 'name'.
    If found and 'forceLoad' is not false, returns it
    If not found or 'forceLoad' is True, evaluates the 'default' function, saves it under the name 'name' and returns it
    """
    if not forceLoad:
        saved = loadObject(name)
    if forceLoad or saved is None:
        saved = default()
        saveObject(saved, name)
    return saved


def deleteCache(name):
    """
    Deletes the cached file under 'name' if exists, otherwise do nothing
    """
    if os.path.isfile(convertFilename(name)):
        os.remove(convertFilename(name))


def checkFolder():
    """
    checks if the cache folder exists, if not otherwise creates it
    """
    if not os.path.exists(os.path.dirname(savedFolder)):
        try:
            os.makedirs(os.path.dirname(savedFolder))
        except OSError as exc:  # Guard against race condition
            import errno
            if exc.errno != errno.EEXIST:
                raise


def convertFilename(name):
    """
    Returns a valid cache path file from the input name
    Replaces characteres not valid for filenames with "_" and appends the extension ".pkl"
    :param name: the name to convert
    :return: the converted path file (valid for cache file)
    """
    return savedFolder + "".join(x if x.isalnum() else "_" for x in name) + ".pkl"


if __name__ == '__main__':
    """
    Tests
    """
    print "Testing functionality:"

    print

    a = 1
    print "saving variable a=1..."
    saveObject(a, "_test_")

    print
    a = -1

    print "loading variable a..."
    a = loadObject("_test_")
    print "a =", a

    print

    print "loading not existing variable..."
    b = loadObject("_thishouldnotexists_")
    print "b = ", b

    print


    def lmb(text):
        print "This was printed from a lambda function! with parameter: ", text
        return "Returned value from lambda function"


    print "loading variable with lambda function..."
    a = cachedObject("_test_", lambda: lmb("should not be shown"))
    print "a =", a

    print

    print "loading not existing variable with lambda function..."
    b = cachedObject("_thishouldnotexists_", lambda: lmb("should be shown"))
    print "b = ", b
    deleteCache("_thishouldnotexists_")

    print

    print "loading variable with lambda function and forceLoad..."
    a = cachedObject("_test_", lambda: lmb("should be shown"), True)
    print "a =", a

    print

    print "Testing decorator"

    deleteCache("_test_")


    @cache_function("_test_")
    def awesomeFunction(a, b):
        print "awesome function evaluated"
        return a * b


    print awesomeFunction(1, 2)

    print awesomeFunction(3, 4)

    deleteCache("_test_")
