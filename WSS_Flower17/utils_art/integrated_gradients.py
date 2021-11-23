import datetime

import numpy as np
import torch
from torch.autograd import Variable


def normal_gradients(outputs, input_x, y, model, device, retain_graph=True):
    return torch.autograd.grad(outputs, input_x, retain_graph=retain_graph)[0]


def integrated_gradients(
        outputs,
        input_x,
        y,
        model,
        device,
        retain_graph=True,
        grad_fn=None,        # gradient function
        baseline=None,
        steps=10):
    """
    Computes integrated gradients for a given network and prediction label.
    
    Integrated gradients is a technique for attributing a deep network's
    prediction to its input features. It was introduced by:
    https://arxiv.org/abs/1703.01365
    
    In addition to the integrated gradients tensor, the method also
    returns some additional debugging information for sanity checking
    the computation. See sanity_check_integrated_gradients for how this
    information is used.
    
    This method only applies to classification networks, i.e., networks
    that predict a probability distribution across two or more class labels.
    
    Access to the specific network is provided to the method via a
    'pred_and_grad_fn' function provided as argument to this method.
    The function takes a batch of inputs and a label, and returns the
    predicted probabilities of the label for the provided inputs, along with
    gradients of the prediction with respect to the input. Such a function
    should be easy to create in most deep learning frameworks.
    
    Args:
      input_x: The specific input for which integrated gradients must be computed.
      target_label_index: Index of the target class for which integrated gradients
        must be computed.
      grad_fn: This is a function that provides access to the
        network's predictions and gradients. It takes the following
        arguments:
        - inputs: A batch of tensors of the same same shape as 'input_x'. The first
            dimension is the batch dimension, and rest of the dimensions coincide
            with that of 'input_x'.
        - target_label_index: The index of the target class for which gradients
          must be obtained.
        and returns:
        - predictions: Predicted probability distribution across all classes
            for each input. It has shape <batch, num_classes> where 'batch' is the
            number of inputs and num_classes is the number of classes for the model.
        - gradients: Gradients of the prediction for the target class (denoted by
            target_label_index) with respect to the inputs. It has the same shape
            as 'inputs'.
      baseline: [optional] The baseline input used in the integrated
        gradients computation. If None (default), the all zero tensor with
        the same shape as the input (i.e., 0*input) is used as the baseline.
        The provided baseline and input must have the same shape.
      steps: [optional] Number of interpolation steps between the baseline
        and the input used in the integrated gradients computation. These
        steps along determine the integral approximation error. By default,
        steps is set to 50.
    
    Returns:
      integrated_gradients: The integrated_gradients of the prediction for the
        provided prediction label to the input. It has the same shape as that of
        the input.
    
      The following output is meant to provide debug information for sanity
      checking the integrated gradients computation.
      See also: sanity_check_integrated_gradients
    
      prediction_trend: The predicted probability distribution across all classes
        for the various (scaled) inputs considered in computing integrated gradients.
        It has shape <steps, num_classes> where 'steps' is the number of integrated
        gradient steps and 'num_classes' is the number of target classes for the
        model.
    """
    if baseline is None:
        baseline = 0*input_x
    assert(baseline.shape == input_x.shape)

    if grad_fn is None: grad_fn = normal_gradients

    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i)/steps)*(input_x-baseline) for i in range(0, steps+1)]

    # error:
    # One of the differentiated Tensors appears to not have been used in the graph.
    # Set allow_unused=True if this is the desired behavior.

    grad_list = []
    for x in scaled_inputs:
        x.cuda(device)
        scores = model({0: x, 1: 1})
        top_scores = scores.gather(1, index=y.unsqueeze(1))
        g = grad_fn(top_scores.mean(), x, y, model, retain_graph)
        grad_list.append(g)
    grads = torch.stack(grad_list)

    # # The following method consumes much memory. Even batch_size=2 will out of memory (22G)
    # # scaled_inputs.shape: [51, 2, 3, 224, 224] # 2 is batch size
    # s_cnt = len(scaled_inputs)
    # x_ = torch.cat(scaled_inputs, dim=0)
    # y_ = y.repeat(s_cnt)
    # scores_ = model({0: x_, 1: 1})
    # top_scores_ = scores_.gather(1, index=y_.unsqueeze(1))
    # grads_ = grad_fn(top_scores_.mean(), x_, y_, model, retain_graph)
    # # x_.shape          : [102, 3, 224, 224]
    # # y_.shape          : [102]
    # # scores_.shape     : [102, 17]
    # # top_scores_.shape : [102, 1]
    # # grads_.shape      : [102, 3, 224, 224]
    # b_size = y.size(0)
    # grads = grads_.split(b_size)
    # grads = torch.stack(grads)
    # # grads.shape : [51, 2, 3, 224, 224]

    # predictions, grads = pred_and_grad_fn(scaled_inputs, target_label_index)
    # shapes: <steps+1>, <steps+1, input_x.shape>
    
    # Use trapezoidal rule to approximate the integral.
    # See Section 4 of the following paper for an accuracy comparison between
    # left, right, and trapezoidal IG approximations:
    # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
    # https://arxiv.org/abs/1908.06214
    grads = grads[:-1] + grads[1:]
    grads = grads / 2.0
    # avg_grads = np.average(grads, axis=0)
    avg_grads = torch.mean(grads, dim=0)
    result_grads = (input_x-baseline)*avg_grads  # shape: <input_x.shape>. result integrated gradients

    # return result_grads, predictions
    return result_grads


def random_baseline_integrated_gradients(
        input_x,
        target_label_index,
        pred_and_grad_fn,
        steps=50,
        num_random_trials=10):
    all_itgr_grads = []
    for i in range(num_random_trials):
        itgr_grads, prediction_trend = integrated_gradients(
            input_x,
            target_label_index=target_label_index,
            pred_and_grad_fn=pred_and_grad_fn,
            baseline=255.0*np.random.random([224, 224, 3]),
            steps=steps)
        all_itgr_grads.append(itgr_grads)
    avg_itgr_grads = np.average(np.array(all_itgr_grads), axis=0)
    return avg_itgr_grads
