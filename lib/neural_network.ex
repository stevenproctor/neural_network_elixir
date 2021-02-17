defmodule NeuralNetwork do
  @moduledoc """
  Documentation for applying a NEURAL NETWORK.
  """

  # Input and Outputs are layers
  # N (where N >= 1) Hidden layers make up a Neural Network
  # Inputs overall are
  #  - weights computed elsewhere, enumeration of N+1 Matrices
  #     - weights are constant across runs of the neural network, independent of input "vector"
  #  - a listing of potential transformation function to be used to normalize between each layer
  #  - running strategy, do we parallelize or sequentially run
  #  - eventually a "vector" of inputs per run of neural network.

  # Weights as constants across multiple runs...
  # Inputs as data/(arguments to "neural network" as "function")
  # Thetas are outputs from each layer...
  # 2 sets of weights required at least for a whole network...
  # Take in a list of transformation functions, one per layer
  # docspecs for sanity of dot_product, layer transformation, etc...

  @type inputs :: [number]
  @type outputs :: [number]
  @type weights :: [number]
  @type layer_weights :: [weights] # Also a neural network layer a.k.a A Matrix!!!
  @type thetas :: [number]
  @type theta_transformation :: (number -> number)
  @type layer_definition :: {layer_weights, theta_transformation}
  @type network_layers :: [layer_definition, layer_definition, ...]
  @type neural_network_runner :: ([weights], inputs -> outputs)
  @type transformations :: [theta_transformation]

  @spec run_neural_network(network_layers, neural_network_runner, inputs) :: outputs
  def run_neural_network(network_layers,
                         layer_runner,
                         starting_inputs) do
    run_layer = fn {layer_weights, theta_transformation}, layer_inputs ->
      layer_runner.(layer_weights, theta_transformation, layer_inputs)
    end
    Enum.reduce(network_layers, starting_inputs, run_layer)
  end

  @spec run_layer_single_process(layer_weights, theta_transformation, inputs) :: outputs
  def run_layer_single_process(layer_weights, theta_transformation \\ &NeuralNetwork.identity/1, inputs) do
    for node_weights <- layer_weights do
      run_layer_node(node_weights, theta_transformation, inputs)
    end
  end

  @spec run_layer_spawn_processes(layer_weights, theta_transformation, inputs) :: outputs
  def run_layer_spawn_processes(layer_weights, theta_transformation \\ &NeuralNetwork.identity/1, inputs) do
    for node_weights <- layer_weights do
      Task.async(fn ->
        run_layer_node(node_weights, theta_transformation, inputs)
      end)
    end
    |> Enum.map(&Task.await/1)
  end

  defp run_layer_node(weights, theta_transformation, inputs) do
    theta_transformation.(dot_product(weights, inputs))
  end

  @spec logit_transform(number) :: number
  def logit_transform(theta) do
    1 / (1 + :math.exp(-theta))
  end

  @spec identity(number) :: number
  def identity(theta) do
    theta
  end

  @spec step_function(number) :: number
  def step_function(number) do
    if number >= 0 do
      1
    else
      0
    end
  end

  @spec rectifier(number) :: number
  def rectifier(number) do
    max(0, number)
  end

  @spec dot_product(weights, inputs) :: number
  def dot_product(weights, inputs) do
    :lists.zipwith(&Kernel.*/2, weights, inputs)
    |> Enum.sum
  end
end
