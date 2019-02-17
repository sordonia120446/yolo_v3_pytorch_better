# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import time

import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO


class Logger(object):

    def __init__(self, log_dir='./nn_logs'):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self._step = 0

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_scalars(self, **kwargs):
        for tag, value in kwargs.items():
            self.scalar_summary(tag, value, self._step)

    def log_named_parameters(self, named_parameters):
        """Parses and logs model.named_parameters()."""
        for tag, value in named_parameters:
            tag = tag.replace('.', '/')
            self.histo_summary(tag, value.data.cpu().numpy(), self._step)
            self.histo_summary(f'{tag}/grad', value.grad.data.cpu().numpy(), self._step)


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))


def savelog(message):
    logging(message)
    with open('savelog.txt', 'a') as f:
        print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message), file=f)
