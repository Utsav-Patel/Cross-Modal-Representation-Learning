import multiprocessing as mp

if mp.cpu_count() >= 10:
    TRAIN_IMAGE_PATH = "/freespace/local/as3503/536/image_features/image_train_embeddings.pkl"
    VALIDATION_IMAGE_PATH = "/freespace/local/as3503/536/image_features/image_val_embeddings.pkl"
    TEST_IMAGE_PATH = "/freespace/local/as3503/536/image_features/image_test_embeddings.pkl"

    TRAIN_TEXT_PATH = "/freespace/local/as3503/536/embeddings_mean/text_title_train_embeddings.pkl"
    VALIDATION_TEXT_PATH = "/freespace/local/as3503/536/embeddings_mean/text_title_val_embeddings.pkl"
    TEST_TEXT_PATH = "/freespace/local/as3503/536/embeddings_mean/text_title_test_embeddings.pkl"

else:
    TRAIN_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/" \
                       "Cross-Modal-Representation-Learning/data/image_features/image_train_embeddings.pkl"
    VALIDATION_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                            "Cross-Modal-Representation-Learning/data/image_features/image_val_embeddings.pkl"
    TEST_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                      "Cross-Modal-Representation-Learning/data/image_features/image_test_embeddings.pkl"

    TRAIN_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                      "Cross-Modal-Representation-Learning/data/embeddings_mean/text_title_train_embeddings.pkl"
    VALIDATION_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/" \
                           "Cross-Modal-Representation-Learning/data/embeddings_mean/text_title_val_embeddings.pkl"
    TEST_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                     "Cross-Modal-Representation-Learning/data/embeddings_mean/text_title_test_embeddings.pkl"
