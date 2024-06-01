class DecoratorUtils:
    @staticmethod
    def iterate_firstdim(func):
        """
        Decorator that allows a function to iterate over the first dimension of a
        tensor.

        Note: The function passed as argument should accept **kwargs so that the
        index of the current iteration can be passed to it without raising error.

        Parameters
        ----------
        func : Callable
            The function to decorate.

        Returns
        -------
        Callable
            The decorated function.
        """

        def wrapper(self, *args, **kwargs):
            for i in range(len(args[0])):
                func(*args, **kwargs, i=i)

        return wrapper
