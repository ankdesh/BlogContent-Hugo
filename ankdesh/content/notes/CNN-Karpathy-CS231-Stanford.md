+++
date = "2016-11-28T10:11:49+05:30"
next = "/notes"
prev = "/notes"
title = "Notes for CS231-Stanford"
toc = true
weight = 5

+++

##### Notes for CNN course by Fei Fei, Andrej Karpathy, Justin Johnson [http://cs231n.stanford.edu/]
Note that these are notes of the course notes. Only notable things I need to remember later are noted in the notes below :P 

[Optimization] (https://docs.google.com/drawings/d/e/2PACX-1vQ8voz0bN0LZXRusAgRESWLxx5cQ0xsOu9xYw7L__FlQe1SlXjkh8wsSF_a4iq45Z9lPq0w005nybau/pub?w=960&amp;h=720)

***
##### [Image classification](http://cs231n.github.io/classification/)

- L1 vs. L2. It is interesting to consider differences between the two metrics. In particular, the L2 distance is much more unforgiving than the L1 distance when it comes to differences between two vectors. That is, the L2 distance prefers many medium disagreements to one big one. Some more details at (http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)

- Validation set size - If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).

- Pactical tips to apply kNN - Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit variance. If the data is very high-dimensional, consider using a dimensionality reduction technique such as PCA. 

- Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be burned on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.

##### [Linear Classification](http://cs231n.github.io/classification/)

- Two functions needed for classification problem - a score function  \\( f: R^D \mapsto R^K  \\)  that maps the raw data to class scores, and a loss function that quantifies the agreement between the predicted scores and the ground truth labels. Classification can then be casted as an optimization problem in which we will minimize the loss function with respect to the parameters of the score function. 

- Analogy of images as high-dimensional points - Since the images are stretched into high-dimensional column vectors, we can interpret each image as a single point in this space. The geometric interpretation of these numbers is that as we change one of the rows of \\(W\\), the corresponding line in the pixel space will rotate in different directions. The biases \\(b\\), on the other hand, allow our classifiers to translate the lines

- Interpretation of linear classifiers as template matching. Another interpretation for the weights \\(W\\) is that each row of \\(W\\) corresponds to a template (or sometimes also called a prototype) for one of the classes. The score of each class for an image is then obtained by comparing each template with the image using an inner product (or dot product) one by one to find the one that "fits" best. With this terminology, the linear classifier is doing template matching, where the templates are learned.

- When would the linear classifier work bad? - For the RGB images (3 "rows" per class in \\(W\\) ), the template matching can only match all 3 color for each pixel (at position p) in input image to the colors in all the rows of \\(W\\) at position p and find which one matches max for the pixel. The average over such competitons for all the pixel will choose the "winner" class. This can be easily fooled in case the the color at the positions is not correct (e.g. we just change the revert the color of an image). Also, in case of Grayscale images, the distinction would be harder for a linear classifier.

- Multiclass Support Vector Machine loss - The SVM loss is set up so that the SVM "wants" the correct class for each image to a have a score higher than the incorrect classes by some fixed margin \\(\\Delta\\). 

- The Multiclass SVM loss for the i-th example is then formalized as follows:
\\( L_i = \\sum_{j\\neq y_i} \\max(0, s_j - s_{y_i} + \\Delta) \\)
where the score for the j-th class is \\( s_j = f(x_i, W)_j \\)

- Note that the max function in SVM loss makes it a non-differentiable function. However the SVM is guaranteed to be a convex fucntion (which makes it easier to optimize as opposed to the non-convex functions). 

- The Multiclass Support Vector Machine "wants" the score of the correct class to be higher than all other scores by at least a margin of delta. If any class has a score higher than the margin, then there will be accumulated loss. Otherwise the loss will be zero. Our objective will be to find the weights that will simultaneously satisfy this constraint for all examples in the training data and give a total loss that is as low as possible.

- L2 Regularization in SVM loss - The most appealing property is that penalizing large weights tends to improve generalization, because it means that no input dimension can have a very large influence on the scores all by itself. For example, suppose that we have some input vector \\(x = [1,1,1,1] \\) and two weight vectors \\(w_1 = [1,0,0,0]\\), \\(w_2 = [0.25,0.25,0.25,0.25] \\). Then \\(w_1^Tx = w_2^Tx = 1\\) so both weight vectors lead to the same dot product, but the L2 penalty of \\(w_1\\) is 1.0 while the L2 penalty of \\(w_2\\) is only 0.25. Therefore, according to the L2 penalty the weight vector \\(w_2\\) would be preferred since it achieves a lower regularization loss. Intuitively, this is because the weights in \\(w_2\\) are smaller and more diffuse. Since the L2 penalty prefers smaller and more diffuse weight vectors, the final classifier is encouraged to take into account all input dimensions to small amounts rather than a few input dimensions and very strongly.

- Softmax classifier - We now interpret the scores as the unnormalized log probabilities for each class and use a cross-entropy loss that has the form: \\(L_i = -\\log\\left(\\frac{e^{f_{y_i}}} \\{ \\sum_j e^{f_j} }\\right) \\)
where we are using the notation \\(f_j\\) to mean the j-th element of the vector of class scores \\(f\\)

- Information theory view for Softmax classifier - The cross-entropy between a "true" distribution \\(p\\) and an estimated distribution \\(q\\) is defined as: \\( H(p,q) = - \\sum_x p(x) \\log q(x) \\)

The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ( \\(q = e^{f_{y_i}} / \\sum_j e^{f_j} \\) as seen above) and the "true" distribution, which in this interpretation is the distribution where all probability mass is on the correct class (i.e. \\(p = [0, \\ldots 1, \\ldots, 0]\\) contains a single 1 at the \\(y_i\\) -th position.).

##### [Optimization: Stochastic Gradient Descent] (http://cs231n.github.io/linear-classify/)

- Gradient can be computed in more than one ways. First is numerical gradient where we calculate the finite differnce of  function over a small change along each axix. Numerical gradient is easy to calculate. The other method is calculating analytical gradient which is error prone to find but is faster to compute.

- Calculating gradient numerically is has a complexity linear in number of parameters. In large classifications problems like for CIFAR-10, (a linear model with 30,730 parameters) the weights are of order [10 X 30,731]. This means that computing gradient using numerical methods would require 30,731 update to calculate the gradient once. This would get worse in deep learning neural networks where the numner of parameters can go up to millions.

- Unlike the numerical gradient, calulating gradient analytically can be more error prone to implement. In practice it is very common to compute the analytic gradient and compare it to the numerical gradient to check the correctness of your implementation.

- ** Mini batch gradient decent ** - In practice, we do not need to calculate the gradient over all the dataset. Just using a small minibatch and calculating gradient over the minibatch gives a good approximation of the gradient over full dataset. This works as the there is a correlation between the training data (we would not have not be able to learn anything if there was none). 
