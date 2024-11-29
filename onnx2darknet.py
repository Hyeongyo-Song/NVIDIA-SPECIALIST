#!/usr/bin/env python3


from collections import OrderedDict
from onnx import numpy_helper
import numpy as np
import onnx
import sys


c_index = 1
w_index = 2
h_index = 3


def make_initial_block(input_info):
    block = OrderedDict()
    block['type'] = 'net'
    block['batch'] = 1
    mapping = {"channels": c_index, "height": h_index, "width": w_index}
    for k, v in mapping.items():
        block[k] = input_info.type.tensor_type.shape.dim[v].dim_value
    return block


def make_route_block(layer):
    block = OrderedDict()
    block['type'] = "route"
    block['layers'] = layer
    return block


def make_conv_block(batchnorm, output_info, attributes):
    block = OrderedDict()
    block['type'] = "convolutional"
    if batchnorm:
        block['batch_normalize'] = 1
    block['filters'] = output_info.type.tensor_type.shape.dim[c_index].dim_value

    for key in ["kernel_shape", "strides", "pads"]:
        if len(set(attributes[key].ints)) != 1:
            raise ValueError(f"Unable to support convolutional block with non-uniform {key}")
    if attributes["dilations"].ints != [1, 1]:
        raise ValueError("Unable to handle dilations")
    mapping = {"size": "kernel_shape", "stride": "strides", "pad": "pads"}
    for k, v in mapping.items():
        block[k] = attributes[v].ints[0]
    block['groups'] = attributes["group"].i
    block['activation'] = "linear"
    return block


def make_relu_block():
    block = OrderedDict()
    block['type'] = "activation"
    block['activation'] = "relu"
    return block


def make_average_pool_block():
    block = OrderedDict()
    block['type'] = "avgpool"
    return block


def make_connected_block(output):
    block = OrderedDict()
    block['type'] = "connected"
    block['output'] = str(output)
    block['activation'] = "linear"
    return block


def onnx2darknet(onnx_model, has_shape=True):
    if not has_shape:
        orig_model = onnx.load(onnx_model)
        model = onnx.shape_inference.infer_shapes(orig_model)
    else:
        model = onnx.load(onnx_model)

    onnx_weights = {}
    weights_count = 0
    for initializer in model.graph.initializer:
        w = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = w
        weights_count += w.size

    onnx_info = {}
    for info in model.graph.value_info:
        onnx_info[info.name] = info

    onnx_nodes = {}
    onnx_nodes_inputs = {}
    onnx_nodes_outputs = {}
    onnx_nodes_attributes = {}
    output_to_index = {}
    for i, node in enumerate(model.graph.node):
        onnx_nodes[i] = node.op_type
        onnx_nodes_inputs[i] = node.input
        onnx_nodes_outputs[i] = node.output
        output_to_index[node.output[0]] = i
        attributes = {}
        for k, attribute in enumerate(node.attribute):
            attributes[attribute.name] = attribute
        onnx_nodes_attributes[i] = attributes

    input_all = model.graph.input
    actual_input = []
    for node in input_all:
        if node.name not in onnx_weights.keys():
            actual_input.append(node)
    if len(actual_input) != 1:
        raise ValueError("Unable to handle multiple inputs")
    output_to_index[actual_input[0].name] = -1
    onnx_nodes_outputs[-1] = [actual_input[0].name]

    blocks = []
    weight_data = []
    block = make_initial_block(actual_input[0])
    blocks.append(block)

    index_to_layer = {}
    index_to_layer[-1] = 0
    last_layer_index = len(model.graph.node) - 1
    add_output_size = {}

    for i, node in enumerate(model.graph.node):
        if onnx_nodes[i] == "Conv":
            input_id = index_to_layer[output_to_index[onnx_nodes_inputs[i][0]]]
            if input_id != len(blocks) - 1:
                block = make_route_block(str(input_id - len(blocks)))
                blocks.append(block)
            index_to_layer[i] = len(blocks)
            batchnorm = False
            if i != last_layer_index and onnx_nodes[i + 1] == "BatchNormalization":
                batchnorm = True
                index_to_layer[i + 1] = len(blocks)
            block = make_conv_block(batchnorm, onnx_info[onnx_nodes_outputs[i][0]], onnx_nodes_attributes[i])
            blocks.append(block)

            if batchnorm:
                batchnorm_weights = [[]]  # to preserve key mapping
                for key in range(1, 5):
                    batchnorm_weights.append(onnx_weights[onnx_nodes_inputs[i + 1][key]])
                weight_data += list(batchnorm_weights[2])  # bias
                weight_data += list(batchnorm_weights[1])  # scale
                weight_data += list(batchnorm_weights[3])  # mean
                weight_data += list(batchnorm_weights[4])  # var
            else:
                weight_data += list(onnx_weights[onnx_nodes_inputs[i][2]])  # bias

            weight_matrix = onnx_weights[onnx_nodes_inputs[i][1]]
            if onnx_nodes_inputs[i][0] in add_output_size.keys():
                old_channels_number = onnx_weights[onnx_nodes_inputs[i][1]].shape[1]
                new_channels_number = add_output_size[onnx_nodes_inputs[i][0]]
                scale_num = int(new_channels_number / old_channels_number)
                weight_matrix = np.repeat(weight_matrix, scale_num, axis=0)

            weight_data += list(np.reshape(weight_matrix, -1))

        elif onnx_nodes[i] == "BatchNormalization":
            continue

        elif onnx_nodes[i] == "Relu":
            index_to_layer[i] = len(blocks)
            block = make_relu_block()
            blocks.append(block)

        elif onnx_nodes[i] == "Add":
            if input_id == len(blocks) - 1:
                raise ValueError("Unable to handle Add layer without subsequent Convolutional layer")
            if onnx_nodes[i + 1] != "Conv":
                raise ValueError("Unable to handle Add layer without subsequent Convolutional layer")
            first_input = onnx_nodes_inputs[i][0]
            first_input_size = 0
            if first_input not in add_output_size.keys():
                first_input_size = onnx_info[first_input].type.tensor_type.shape.dim[c_index].dim_value
            else:
                first_input_size = add_output_size[first_input]
            first_input_id = index_to_layer[output_to_index[first_input]]

            if first_input_id != len(blocks) - 1:
                block = make_route_block(str(first_input_id - len(blocks)))
                blocks.append(block)

            skipper = 0
            second_input = onnx_nodes_inputs[i][1]
            second_input_size = 0
            if second_input not in add_output_size.keys():
                second_input_size = onnx_info[second_input].type.tensor_type.shape.dim[c_index].dim_value
            else:
                second_input_size = add_output_size[second_input]
            second_input_id = index_to_layer[output_to_index[second_input]]

            if second_input_id != len(blocks) - 1:
                block = make_route_block(str(second_input_id - len(blocks)))
                blocks.append(block)
                skipper = 1

            index_to_layer[i] = len(blocks)
            block = make_route_block("-1, " + str(-skipper - 1))
            blocks.append(block)
            add_output_size[onnx_nodes_outputs[i][0]] = first_input_size + second_input_size

        elif onnx_nodes[i] == "AveragePool":
            index_to_layer[i] = len(blocks)
            block = make_average_pool_block()
            blocks.append(block)

        elif onnx_nodes[i] == "Gemm":
            index_to_layer[i] = len(blocks)
            block = make_connected_block(onnx_weights[onnx_nodes_inputs[i][2]].size)
            blocks.append(block)

            weight_data += list(onnx_weights[onnx_nodes_inputs[i][2]])  # bias
            weight_data += list(np.reshape(onnx_weights[onnx_nodes_inputs[i][1]], -1))  # weight

        elif onnx_nodes[i] == "Identity":
            onnx_weights[onnx_nodes_outputs[i][0]] = onnx_weights[onnx_nodes_inputs[i][0]]

        else:
            print(f"Not handling {onnx_nodes[i]}")

    return blocks, np.array(weight_data)


def save_cfg(blocks, cfgfile):
    with open(cfgfile, 'w') as fp:
        for block in blocks:
            fp.write('[%s]\n' % (block['type']))
            for key, value in block.items():
                if key != 'type':
                    fp.write('%s=%s\n' % (key, value))
            fp.write('\n')


def save_weights(data, weightfile):
    wsize = data.size
    weights = np.zeros((wsize + 4,), dtype=np.int32)
    weights[0] = 0      # major
    weights[1] = 1      # minor
    weights[2] = 0      # revision
    weights[3] = 0      # net.seen
    weights.tofile(weightfile)
    weights = np.fromfile(weightfile, dtype=np.float32)
    weights[4:] = data
    weights.tofile(weightfile)


def main():
    filename = sys.argv[1]
    cfg, weight = onnx2darknet(filename, False)
    root_name = '.'.join(filename.split('.')[:-1])
    save_cfg(cfg, f"{root_name}.cfg")
    save_weights(weight, f"{root_name}.weights")


if __name__ == '__main__':
    main()
