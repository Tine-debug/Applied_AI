


# Task 2.2.1


def Test_k_values(number_folds):
    # Split data into three parts
    fold_size = int(np.floor(Ltr_set.size / number_folds))
    folds = []
    for i in range (0, number_folds-1): 
        folds.append((Tr_set[i*fold_size:((i+1)*fold_size):1], Ltr_set[i*fold_size:((i+1)*fold_size):1]))
    folds.append((Tr_set[(number_folds-1)*fold_size: Ltr_set.size:1], Ltr_set[(number_folds-1)*fold_size: Ltr_set.size:1]))

    #get accuracy for each k
    average_accuracy = np.zeros(20)
    for k in range (1, 21):
        accuracy_folds = np.zeros(number_folds)

        
    #run training/tests for all the folds and calulate accuracy for each
        for i in range(0,number_folds):
            training_data = np.empty((0, folds[0][0].shape[1]))
            training_data_labels = np.empty((0,))

            
            for j in range(0 ,number_folds):
                if (j == i):
                    test_data = folds[i][0]
                    test_data_labels = folds [i][1]
                else:
                    training_data = np.append(training_data, folds[j][0], axis=0)
                    training_data_labels = np.append(training_data_labels, folds[j][1], axis = 0)

            
            Labels_predicted_L2 = predict_L2(test_data, k, training_data, training_data_labels)
            accuracy_folds[i] = np.mean(Labels_predicted_L2 == test_data_labels)

            
        average_accuracy[k-1] = np.average(accuracy_folds)
    return np.argmax(average_accuracy)+1
            


print(Test_k_values(3))

# The best k is 3


## Task 2.2.2

Labels_predicted_L2=predict_L2(Test_images, 3, Tr_set, Ltr_set)
print("Accuracy:", np.mean(Labels_predicted_L2==L_test))

#Accuracy: 0.8189
