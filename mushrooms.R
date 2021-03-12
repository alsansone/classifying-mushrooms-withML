FLAGS <-flags(
  flag_numeric("nodes", 128),
  flag_numeric("batch_size", 100),
  flag_string("activation", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 30))

model <- keras_model_sequential()
model %>%
  layer_dense(units = FLAGS$nodes, input_shape = dim(mushrooms_train_one_hot)[2], activation = FLAGS$activation) %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation) %>%
  layer_dense(units = 2, activation = 'sigmoid')

model %>% compile(
  optimizer = optimizer_adam(lr = FLAGS$learning_rate),
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
)

history <- model %>% fit(
  as.matrix(mushrooms_train_one_hot),
  as.matrix(mushrooms_train_labels),
  epochs = FLAGS$epochs,
  batch_size = FLAGS$batch_size,
  verbose = 0,
  validation_data = list(as.matrix(mushrooms_val_one_hot), as.matrix(mushrooms_val_labels)))
