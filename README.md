
The problem come from Andrew Ng's machine learning course projects from Coursera, and 
I'd like to implement them in python instead of matlab/octave

For the first part of the exercise, we are asked to implement a classification model based on students' scores from
two exams and wheather the student was accepted to the school or not.
For each training example, you have the applicant’s scores on two exams and the admissions
decision.  The model will be used to predict future student's probability of getting
admitted given the results of two exams.
Sigmoid function is used as activation function.
I implemented cost function and gradient function as input for the minimize from
scipy library to get optimized parameters, and in turn the parameters are used to
predict new student's admission probability.


DO NOT USE THIS SOURCE CODE FOR THE EXERCISES/PROJECTS IN COURSERA MACHINE
 LEARNING COURSE.