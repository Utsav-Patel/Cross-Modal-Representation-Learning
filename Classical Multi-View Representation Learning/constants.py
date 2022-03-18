import multiprocessing as mp

TEXT_ELEMENT = "title"
TYPE_EMBEDDING = "image"

if mp.cpu_count() >= 10:
    TRAIN_IMAGE_PATH = "/freespace/local/as3503/536/entire/image_features/image_train_embeddings.pkl"
    VALIDATION_IMAGE_PATH = "/freespace/local/as3503/536/entire/image_features/image_val_embeddings.pkl"
    TEST_IMAGE_PATH = "/freespace/local/as3503/536/entire/image_features/image_test_embeddings.pkl"

    TRAIN_TEXT_PATH = "/freespace/local/as3503/536/entire/embeddings_mean/text_" + TEXT_ELEMENT + "_train_embeddings.pkl"
    VALIDATION_TEXT_PATH = "/freespace/local/as3503/536/entire/embeddings_mean/text_" + TEXT_ELEMENT + "_val_embeddings.pkl"
    TEST_TEXT_PATH = "/freespace/local/as3503/536/entire/embeddings_mean/text_" + TEXT_ELEMENT + "_test_embeddings.pkl"

else:
    TRAIN_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/" \
                       "Cross-Modal-Representation-Learning/data/entire/image_features/image_train_embeddings.pkl"
    VALIDATION_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                            "Cross-Modal-Representation-Learning/data/entire/image_features/image_val_embeddings.pkl"
    TEST_IMAGE_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                      "Cross-Modal-Representation-Learning/data/entire/image_features/image_test_embeddings.pkl"

    TRAIN_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                      "Cross-Modal-Representation-Learning/data/entire/embeddings_mean/text_" + TEXT_ELEMENT +\
                      "_train_embeddings.pkl"
    VALIDATION_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/" \
                           "Cross-Modal-Representation-Learning/data/entire/embeddings_mean/text_" + TEXT_ELEMENT +\
                           "_val_embeddings.pkl"
    TEST_TEXT_PATH = "D:/My_Files/Rutgers/Courses/536 - Machine Learning/Project/"\
                     "Cross-Modal-Representation-Learning/data/entire/embeddings_mean/text_" + TEXT_ELEMENT +\
                     "_test_embeddings.pkl"
