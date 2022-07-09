class DataConversion:

    def create_result_array(self, acc_entity):
        array = []
        array1 = []

        # storing in array
        array.append(acc_entity.filename)
        array.append(acc_entity.score)
        array.append(acc_entity.val_score)
        array1.append(array)

        return array1

    def create_submission_array(self, sub_entity):

        array1 = []
        y_pred = sub_entity.predictions
        id_ = sub_entity.id_
        id_2 = sub_entity.id_2
        id_3 = sub_entity.id_3

        for i in range(len(y_pred)):
            array = []
            # storing in array
            if id_:
                array.append(id_[i])

            if id_2:
                array.append(id_2[i])

            if id_3:
                array.append(id_3[i])

            array.append(y_pred[i])

            array1.append(array)

        return array1
