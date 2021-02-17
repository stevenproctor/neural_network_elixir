defmodule NeuralNetworkRunner do
  @external_resource layers_path = Path.join([__DIR__, "layers.erl"])

  {:ok, layers} = :file.consult(layers_path)
  @layers layers
  @layers_count length(layers)
  if (@layers_count < 2) do
    raise "Invalid Neural Network Definition: Neural Network definition from #{layers_path} needs at least 2 layers defined"
  end

  @transforms List.duplicate(&NeuralNetwork.logit_transform/1, @layers_count)
  @layer_definitions List.zip([@layers, @transforms])

  @spec run_network(NeuralNetwork.inputs) :: NeuralNetwork.outputs
  def run_network(inputs) do
    task = Task.async(fn ->
      NeuralNetwork.run_neural_network(@layer_definitions,
                                       &NeuralNetwork.run_layer_spawn_processes/3,
                                       # &NeuralNetwork.run_layer_single_process/3,
                                       inputs)
    end)
    Task.await(task)
  end

end
