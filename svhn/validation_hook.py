import tensorflow as tf
import numpy as np

class ValidationHook(tf.train.SessionRunHook):
    """Runs evaluation of a given estimator, at most every N steps.

    Can do early stopping on validation metrics if `early_stopping_rounds` is
    provided.
    """
    def __init__(self, estimator, input_fn, every_n_steps=None,
                 early_stopping_rounds=None, early_stopping_metric="loss",
                 early_stopping_metric_minimize=True):
        """Initializes a `ValidationHook`.
        Args:
            estimator: `tf.train.Estimator` object: The estimator which is to be
                used to evaluate on the validation data.
            input_fn: A function that constructs the input validation data for
                evaluation.
                See @{$get_started/premade_estimators#create_input_functions} for
                more information. The function should construct and return one of
                the following:

                * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
                    tuple (features, labels) with same constraints as below.
                * A tuple (features, labels): Where `features` is a `Tensor` or a
                    dictionary of string feature name to `Tensor` and `labels` is a
                    `Tensor` or a dictionary of string label name to `Tensor`. Both
                    `features` and `labels` are consumed by `model_fn`. They should
                    satisfy the expectation of `model_fn` from inputs.

            every_n_steps: `int`, evaluate the estimator every N steps.
            early_stopping_rounds: `int`. If the metric indicated by
                `early_stopping_metric` does not change according to
                `early_stopping_metric_minimize` for this many steps, then training
                will be stopped.
            early_stopping_metric: `string`, name of the metric to check for early
                stopping.
            early_stopping_metric_minimize: `bool`, True if `early_stopping_metric` is
                expected to decrease (thus early stopping occurs when this metric
                stops decreasing), False if `early_stopping_metric` is expected to
                increase. Typically, `early_stopping_metric_minimize` is True for
                loss metrics like mean squared error, and False for performance
                metrics like accuracy.
        Raises:
        ValueError: if `every_n_steps` is non-positive.
        ValueError: if `early_stopping_rounds` is non-positive.
        """
        if every_n_steps is not None and every_n_steps<= 0:
            raise ValueError("invalid every_n_steps={}.".format(every_n_steps))
        if early_stopping_rounds is not None and every_n_steps <= 0:
            raise ValueError("invalid early_stopping_rounds={}.".format(
                early_stopping_rounds))

        self._iter_count = 0
        self._estimator = estimator
        self._input_fn = input_fn
        self._timer = tf.train.SecondOrStepTimer(None, every_n_steps)
        self._should_trigger = False

        self._early_stopping_iter_count = 0
        self._early_stopping_rounds = early_stopping_rounds
        self._early_stopping_metric = early_stopping_metric
        self._early_stopping_metric_minimize = early_stopping_metric_minimize

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self._early_stopping_iter_count = 0
        if self._early_stopping_metric_minimize:
            self._prev_metric_value = 1e7
        else:
            self._prev_metric_value = 0

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            output = self._estimator.evaluate(self._input_fn)

            # Logging output
            original = np.get_printoptions()
            np.set_printoptions(suppress=True)
            tf.logging.info("%s", ", ".join(["val_{} = {}".format(tag,
                output[tag]) for tag in output]))
            np.set_printoptions(**original)

            if self._early_stopping_rounds is not None:
                curr_value = output[self._early_stopping_metric]
                if (self._early_stopping_metric_minimize and\
                curr_value > self._prev_metric_value) or\
                (not self._early_stopping_metric and\
                curr_value < self._prev_metric_value):
                    self._early_stopping_iter_count += 1

                if self._early_stopping_iter_count ==\
                self._early_stopping_rounds:
                    run_context.request_stop()
                self._prev_metric_value = curr_value

            self._timer.update_last_triggered_step(self._iter_count)
        self._iter_count += 1
