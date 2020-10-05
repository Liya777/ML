# Support Vector Machine
Support vector machine is used to solve supervised machine learning problems, both classification, and regression. The classes can be separated by a line known as the 
line(2D)/hyperplane(3D). The criteria for positioning the hyperplane is that on the creation of a hyperplane, two margins are also created such that they lie on either 
side of hyperplane, linearly separating the two classes and also pass through the nearest point in the respective class. The dotted line in the figure represents the margin. 
The distance between the two margins is known as the marginal distance. Support vectors are the points that pass through the margins. The significance of the margins is that 
it classifies the points into two classes. The best classification results are obtained when we make the model more generalized. This can be obtained by maximizing the 
marginal distance. Hence we can say that hyperplane and margin are to be located in such a way that the marginal distance is maximum. This is in the case of linearly separable.
Consider a two-dimensional graph. In order to solve non linearly separable, the support vector machine uses a technique called SVM kernels. SVM kernels convert low dimension(2D) 
to high dimension(3D). The separation of the classes is clearly visible and a hyperplane is created between them. This is a brief explanation about the basics of the support 
vector machine.

# K NEAREST NEIGHBOUR
KNN is used to classify both classification and regression problems. KNN is mainly used for classifying nonlinear data sets; which can not be separated using a single line.
Basically, we need to find the k value. K value means the number of nearest neighboring points that are to be considered for classification. They are selected based on the 
distance. The distance is calculated by Euclidean distance and  Manhattan distance. 

Euclidean distance: This is basically calculated using Pythagoras theorem. 
Manhattan distance: Manhattan distance is basically calculated by constructing a right triangle. 
Thus, these two methods help us to find the value of k.

Classification use case:
Suppose k=5, which means we obtained 5 points in the dataset. Suppose, in the classification dataset, 3 point belongs to class A while the other two points belong to class B, 
we can conclude that, the new point that is to be classified belongs to class A since the k value of class A is greater than class B.
The k value should be odd as it should not produce equal distribution in both the classes.

Regression use case:
In a regression use case, there won't be any classes. Suppose the k value to be 5. The average mean of the nearest 5 points is calculated and assigned to the new point for 
which the value is to be predicted. 
