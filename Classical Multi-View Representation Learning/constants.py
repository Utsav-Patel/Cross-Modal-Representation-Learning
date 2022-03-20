import multiprocessing as mp

TEXT_ELEMENT = "title"
TYPE_EMBEDDING = "text"

if mp.cpu_count() >= 10:
    # TRAIN_IMAGE_PATH = "/freespace/local/as3503/536/image_features/image_train_embeddings.pkl"
    # VALIDATION_IMAGE_PATH = "/freespace/local/as3503/536/image_features/image_val_embeddings.pkl"
    # TEST_IMAGE_PATH = "/freespace/local/as3503/536/image_features/image_test_embeddings.pkl"

    TRAIN_IMAGE_PATH = "/common/home/upp10/Desktop/Cross-Modal-Representation-Learning/data/professor/embeddings_train1.pkl"
    VALIDATION_IMAGE_PATH = "/common/home/upp10/Desktop/Cross-Modal-Representation-Learning/data/professor/embeddings_val1.pkl"
    TEST_IMAGE_PATH = "/common/home/upp10/Desktop/Cross-Modal-Representation-Learning/data/professor/embeddings_test1.pkl"

    # TRAIN_TEXT_PATH = "/freespace/local/as3503/536/embeddings_all_means/text_" + TEXT_ELEMENT + "_train_embeddings.pkl"
    # VALIDATION_TEXT_PATH = "/freespace/local/as3503/536/embeddings_all_means/text_" + TEXT_ELEMENT + "_val_embeddings.pkl"
    # TEST_TEXT_PATH = "/freespace/local/as3503/536/embeddings_all_means/text_" + TEXT_ELEMENT + "_test_embeddings.pkl"

    TRAIN_TEXT_PATH = "/common/home/upp10/Desktop/Cross-Modal-Representation-Learning/data/professor/" + TEXT_ELEMENT + "_embeddings_train.pkl"
    VALIDATION_TEXT_PATH = "/common/home/upp10/Desktop/Cross-Modal-Representation-Learning/data/professor/" + TEXT_ELEMENT + "_embeddings_val.pkl"
    TEST_TEXT_PATH = "/common/home/upp10/Desktop/Cross-Modal-Representation-Learning/data/professor/" + TEXT_ELEMENT + "_embeddings_test.pkl"

else:
    TRAIN_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/" \
                       "Cross-Modal-Representation-Learning/data/professor/embeddings_train1.pkl"
    VALIDATION_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                            "Cross-Modal-Representation-Learning/data/professor/embeddings_val1.pkl"
    TEST_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                      "Cross-Modal-Representation-Learning/data/professor/embeddings_test1.pkl"

    TRAIN_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                      "Cross-Modal-Representation-Learning/data/professor/" + TEXT_ELEMENT +\
                      "_embeddings_train.pkl"
    VALIDATION_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/" \
                           "Cross-Modal-Representation-Learning/data/professor/" + TEXT_ELEMENT +\
                           "_embeddings_val.pkl"
    TEST_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                     "Cross-Modal-Representation-Learning/data/professor/" + TEXT_ELEMENT +\
                     "_embeddings_test.pkl"
