from data_loader.weighted_data_loader import loadCIFARData, getWeightedDataLoaders, create_clean_initial_weights
from weight_samples.visual_similarity import train as train_visual_embedding
from coefficient_update import train as train_coefficient_update
from utils import load_config
from ptdarts import augment
from weight_samples import update_similarity_weights, train

def main():
    # load data
    #create_clean_initial_weights('./data/', 'cifar-10-batches-py')

    train_data, val_data, test_data = loadCIFARData()
    print(train_data)
    train_queue, val_queue, test_loader = getWeightedDataLoaders(train_data, val_data, test_data)
    # First Stage: calculate network weights W1 with fixed architectiure A by minimizing training loss,
    # then apply to validation set and see how it performs

    genotype = "Genotype(normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('skip_connect', 0), ('dil_conv_3x3', 2)], " \
                   "[('sep_conv_3x3', 1), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('skip_connect', 0)]]," \
                   "normal_concat=range(2, 6)," \
                   "reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)]," \
                   "[('skip_connect', 3), ('max_pool_3x3', 0)], [('skip_connect', 2), ('max_pool_3x3', 0)]]," \
                   "reduce_concat=range(2, 6))"
    w_config = load_config('weight_samples/config.yml')
    in_size = train_data[0][0].shape[1]
    #model = augment.main(in_size, train_queue, val_queue, genotype, weight_samples=False)

    # Use validation performance to re-weight each training example with three scores
    # for each training sample and update them in instance_weights.npy

    #update_similarity_weights.calculate_similarity_weights(train_data, train_queue, model, val_queue, w_config)

    # Second Stage: based on the calculated weights for each training instance, calculates a second
    # set of weights given the DARTS architecture by minimizing weighted training loss
    model = augment.main(in_size, train_queue, val_queue, genotype, weight_samples=True)

    # Third Stage.1: based on the new set of weights, update the architecture A by minimizing the validation loss
    #search.main(train_queue, val_queue)

    # Third Stage.2: update image embedding V by minimizing the validation loss
    vis_config = load_config('weight_samples/visual_similarity/config.yml')
    train_visual_embedding.train(train_queue, val_queue, vis_config['learning_rate'], vis_config['epochs'])

    # Third Stage.3: update coefficient vector r by minimizing the validation loss
    # Given the learned Architecture and image embedding, do a linear regression to obtain the coefficient vector r
    coeff_config = load_config('./coefficient_update/config.yml')
    train_coefficient_update.train(train_queue, val_queue, coeff_config['learning_rate'], coeff_config['epochs'])

if __name__ == "__main__":
    main()