"""
Implementation of cache to functions.

normal usage:

@cache_function("foo_{0}")
def foo(bar):
    #something

"""
import inspect
import os
import pickle
from functools import wraps

savedFolder = "_cache_/"

# Constants
TYPE_NORMAL = [False, False]
TYPE_ALWAYSEXECUTE = [True, False]
TYPE_DONTSAVE = [False, True]
TYPE_DISABLE = [True, True]


def cache_function(name, type=lambda *args, **kwargs: TYPE_NORMAL, timestamp=0):
    """
    Decorator to mark functions as cached under a name. (Memoization)
    The name is formatted {.format()} with the function parameters, to allow parameters-dependent caches
    The type is a function which is passed the function parameters, to allow changing the behaviour under some conditions
    - TYPE_NORMAL:
        If the name is already cached, the value is returned without evaluating the function.
        If the name is not cached, the function is evaluated, saved and returned
    - TYPE_ALWAYSEXECUTE:
        The function is evaluated, saved and returned (either if cache exists or not)
    - TYPE_DONTSAVE:
        If the name is already cached, the value is returned without evaluating the function.
        If the name is not cached, the function is evaluated and returned, but not saved
    - TYPE_DISABLE:
        The function is evaluated and returned. Existing cache, if any, is not modified. (Same as removing the decorator)

    Examples:   @cache_function("mycache") will store the cache under 'mycache'
                @cache_function("{0}_cache") will store foo(1) under '1_cache', foo('s') under 's_cache', and so on.
                @cache_function("mycache", lambda x: cache_function.TYPE_ALWAYSEXECUTE if x<0 else cache_function.TYPE_NORMAL) will always execute the function if the first parameter of the function is negative
    """

    def func_decorator(func):
        argspec = inspect.getargspec(func)
        positional_count = len(argspec.args) - len(argspec.defaults)
        defaults = dict(zip(argspec.args[positional_count:], argspec.defaults))

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            newargs = args[:positional_count]
            newkwargs = defaults.copy()
            newkwargs.update({k: v for k, v in zip(argspec.args[positional_count:], args[positional_count:])})
            newkwargs.update(kwargs)

            args = newargs
            kwargs = newkwargs

            forceLoad, dontSave = type(*args, **kwargs)
            return cachedObject(name.format(*args, **kwargs), lambda: func(*args, **kwargs), forceLoad, dontSave, timestamp)

        return func_wrapper

    return func_decorator


cache_function.TYPE_NORMAL = TYPE_NORMAL
cache_function.TYPE_ALWAYS_EXECUTE = TYPE_ALWAYSEXECUTE
cache_function.TYPE_DONTSAVE = TYPE_DONTSAVE
cache_function.TYPE_DISABLE = TYPE_DISABLE


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


def cachedObject(name, default=lambda: None, forceLoad=False, dontSave=False, timestamp=0):
    """
    Implements a cached system.
    Tries to load the cached object with name 'name'.
    If found and 'forceLoad' is not false, returns it
    If not found or 'forceLoad' is True or timestamp is different than saved one, evaluates the 'default' function, saves it under the name 'name'  unless dontSave is true, and returns it
    """
    if not forceLoad:
        data = loadObject(name)
        saved, stamp = data if data is not None else (None, None)
    if forceLoad or saved is None or timestamp != stamp:
        saved = default()
        if not dontSave:
            saveObject((saved, timestamp), name)
    return saved


def deleteObject(name):
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
    deleteObject("_thishouldnotexists_")

    print

    print "loading variable with lambda function and forceLoad..."
    a = cachedObject("_test_", lambda: lmb("should be shown"), True)
    print "a =", a

    print

    print "Testing decorator"

    deleteObject("_test_")


    @cache_function("_test_", lambda a, b: a == 3)
    def awesomeFunction(a, b):
        print "awesome function evaluated"
        return a * b


    print awesomeFunction(1, 2)

    print awesomeFunction(3, 4)

    deleteObject("_test_")
