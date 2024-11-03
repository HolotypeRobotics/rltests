import numpy as np

class PROControlModel:
    def __init__(self, stimuli_size, num_responses, base_learning_rate=0.01, td_discount=0.95, 
                 proactive_control_scaling=0.5, mutual_inhibition_scaling=0.5, 
                 reactive_control_scaling=0.25, noise_variance=0.05, 
                 time_constant=0.05, excitation_scaling=0.5, 
                 inhibition_threshold=0.5, hardwired_response_scaling=0.5, 
                 negative_valence=1.0, action_threshold=0.5):
        """
        Initialize the PRO-control model.

        Args:
            stimuli_size (int): The number of stimuli in the task.
            num_responses (int): The number of possible responses.
            learning_rate (float): The learning rate for updating weights.
            td_discount (float): The discount factor for temporal difference learning.
            proactive_control_scaling (float): The scaling factor for proactive control signals.
            mutual_inhibition_scaling (float): The scaling factor for mutual inhibition between response units.
            reactive_control_scaling (float): The scaling factor for reactive control signals.
            noise_variance (float): The variance of the noise added to the unit response.
            time_constant (float): The time constant for updating the unit response.
            excitation_scaling (float): The scaling factor for excitation signals.
            inhibition_threshold (float): The threshold for triggering proactive control.
            hardwired_response_scaling (float): The scaling factor for hardwired response weights.
            negative_valence (float): The negative valence associated with a bad outcome.
            action_threshold (float): The threshold for triggering an action.
        """
        self.stimuli_size = stimuli_size
        self.num_responses = num_responses
        self.base_learning_rate = base_learning_rate
        self.td_discount = td_discount
        self.top_down_control_scaling = proactive_control_scaling
        self.mutual_inhibition_scaling = mutual_inhibition_scaling
        self.reactive_control_scaling = reactive_control_scaling
        self.noise_variance = noise_variance
        self.time_constant = time_constant
        self.excitation_scaling = excitation_scaling
        self.inhibition_threshold = inhibition_threshold
        self.hardwired_response_scaling = hardwired_response_scaling
        self.negative_valence = negative_valence
        self.action_threshold = action_threshold

        # Initialize Weights
        # i: response size
        # j: stimuli size
        self.w_responseOutcomeConj = np.zeros((self.num_responses, self.stimuli_size))  #(Ws) R-O conjunction weights, map D to S (i x j)
        
        # Critic weights
        self.w_valuePredictions = np.zeros((self.num_responses, self.stimuli_size))  #(U) Value prediction weights,X to V (i x j x k)
        self.w_inhibitoryResponses = np.zeros((self.num_responses, self.stimuli_size))  #(Wi) Inhibitory response weights map C to I (i x j)
        self.w_reactiveControl = np.zeros((self.num_responses, self.stimuli_size))  #(Ww) Reactive control weights map wN to E/I (i x j)
        self.w_mutualInhibition = np.zeros((self.num_responses, self.num_responses))  #(Wi) FIXED Mutual inhibition weights (i x j)
        self.w_proactiveControl = np.zeros((self.num_responses, self.stimuli_size))  #(Wf) Proactive control weights map S to E/I (i x k)

        # Actor weights
        self.w_hardwiredResponses = np.zeros((self.num_responses, self.stimuli_size))  #(Wc) FIXED response weights map D to E (i x k)
    
        # Initialize other variables
        self.delay_chain = np.zeros((self.stimuli_size, self.num_responses))  # Temporal representation of stimuli
        self.previous_response = np.zeros(self.num_responses)
        self.positiveSurprize = 0  # Positive surprise signal
        self.negativeSurprize = 0  # Negative surprise signal
        self.learning_rate = np.full(self.num_responses, self.base_learning_rate)  # Learning rate for each response

    # Function to calculate the response-outcome conjunction
    def responseOutcomeConj(self, stimuli):
        """
        Calculate the response-outcome conjunction.

        Args:
            stimuli (numpy.ndarray): The current stimuli vector.

        Returns:
            numpy.ndarray: The predicted response-outcome conjunction vector.
        """
        return np.dot(stimuli, self.w_responseOutcomeConj)

    # Function to learn response-outcome weights
    def learnResponseOutcomeWeights(self, actual_roc, predicted_roc, stimuli, behaviorally_relevant=False):
        """
        Update the response-outcome weights based on the actual and predicted response-outcome conjunctions.

        Args:
            actual_roc (numpy.ndarray): The actual response-outcome conjunction vector.
            predicted_roc (numpy.ndarray): The predicted response-outcome conjunction vector.
            stimuli (numpy.ndarray): The current stimuli vector.
            behaviorally_relevant (bool): Whether the current trial is behaviorally relevant.
        """
        if behaviorally_relevant:
            self.w_responseOutcomeConj += self.learning_rate * ((self.negative_valence * actual_roc) - predicted_roc) * stimuli

    # Function to calculate the temporal difference error vector
    def TDErrorVec(self, reward_vec, pred_future_value_vec, pre_current_value_vec):
        """
        Calculate the temporal difference error vector.

        Args:
            reward_vec (numpy.ndarray): The reward vector for the current trial.
            pred_future_value_vec (numpy.ndarray): The predicted value vector for the next trial.
            pre_current_value_vec (numpy.ndarray): The predicted value vector for the current trial.

        Returns:
            numpy.ndarray: The temporal difference error vector.
        """
        return reward_vec + (self.td_discount * pred_future_value_vec) - pre_current_value_vec

    # Function to predict values
    def predictValues(self, delay_chain):
        """
        Predict the value of each response based on the delay chain representation.

        Args:
            delay_chain (numpy.ndarray): The delay chain representation of the stimuli.

        Returns:
            numpy.ndarray: The predicted value vector.
        """
        return np.dot(delay_chain, self.w_valuePredictions)

    def updateLearningRate(self):
        """
        Update the learning rate based on the surprise signals.
        """
        self.learning_rate = self.base_learning_rate / (1 + (self.negativeSurprize + self.positiveSurprize))

    def excitation(self, stimuli, response_outcome_conj):
        """
        Calculate the excitation signal for each response unit.

        Args:
            stimuli (numpy.ndarray): The current stimuli vector.
            response_outcome_conj (numpy.ndarray): The predicted response-outcome conjunction vector.

        Returns:
            numpy.ndarray: The excitation signal.
        """
        proactive_control = np.dot((-response_outcome_conj), self.w_proactiveControl)
        proactive_control = np.maximum(0, proactive_control)
        reactive_control = np.sum(self.w_reactiveControl, axis=1)
        reactive_control = np.maximum(0, reactive_control)
        return self.excitation_scaling * (np.dot(stimuli, self.w_hardwiredResponses) + proactive_control + reactive_control)
        
    # Function to calculate the inhibition
    def inhibition(self, response, response_outcome_conj):
        """
        Calculate the inhibition signal for each response unit.

        Args:
            response (numpy.ndarray): The current response vector.
            response_outcome_conj (numpy.ndarray): The predicted response-outcome conjunction vector.

        Returns:
            numpy.ndarray: The inhibition signal.
        """
        proactive_control_signals = np.dot(self.w_proactiveControl, response_outcome_conj)
        proactive_control_signals = np.maximum(0, proactive_control_signals)

        reactive_control_signals = np.sum(self.w_reactiveControl, axis=1)
        reactive_control_signals = np.maximum(0, reactive_control_signals)

        mutual_inhibition_signals = self.mutual_inhibition_scaling * (np.dot(response, self.w_mutualInhibition))
        
        return mutual_inhibition_signals + self.top_down_control_scaling * (proactive_control_signals + reactive_control_signals)

    # Function to calculate the unit response for the actor
    def unitResponse(self, stimuli):
        """
        Calculate the response of each unit based on the stimuli, previous response, and inhibition signals.

        Args:
            stimuli (numpy.ndarray): The current stimuli vector.
            previous_response (numpy.ndarray): The previous response vector.

        Returns:
            numpy.ndarray: The unit response vector.
        """
        inhibition = self.inhibition(self.previous_response,self.responseOutcomeConj(stimuli) )
        noise = np.random.normal(0, self.noise_variance)
        return self.previous_response + self.time_constant * (self.excitation(stimuli) * (1 - self.previous_response) - (self.previous_response + .05) * (inhibition + 1) + noise)

    # Function to learn wNToResponseUnits weights
    def learnReactiveControlWeights(self, valence_signal, action_was_executed):
        """
        Update the reactive control weights based on the valence of the outcome and whether an action was executed.

        Args:
            valence_signals (numpy.ndarray): The valence signals associated with each response.
            action_was_executed (bool): Whether an action was executed in the current trial.
        """
        if action_was_executed:
            self.w_reactiveControl += self.reactive_control_scaling * (self.w_reactiveControl + (valence_signal * self.negativeSurprize))

        # Constrain the reactive control weights to be between -1 and 1
        self.w_reactiveControl = np.clip(self.w_reactiveControl, -1, 1)

    # Function to calculate surprise
    def calculateSurprise(self, outcome, prediction):
        """
        Calculate the surprise signal for each response unit.

        Args:
            outcome (numpy.ndarray): The actual outcome vector.
            prediction (numpy.ndarray): The predicted outcome vector.

        Returns:
            numpy.ndarray: The positive surprise signal.
            numpy.ndarray: The negative surprise signal.
        """
        self.negativeSurprize = np.sum(outcome - prediction)
        self.positiveSurprize = np.sum(prediction - outcome)
        return self.positiveSurprize, self.negativeSurprize

    # Function to update value predictions (Critic)
    def updateValuePredictions(self, reward_vec, predicted_values, delay_chain):
        """
        Update the value prediction weights based on the temporal difference error.

        Args:
            reward_vec (numpy.ndarray): The reward vector for the current trial.
            predicted_values (numpy.ndarray): The predicted value vector for the current trial.
            delay_chain (numpy.ndarray): The delay chain representation of the stimuli.
        """
        td_error = self.TDErrorVec(reward_vec, predicted_values, self.predictValues(delay_chain))
        self.w_valuePredictions += self.learning_rate * td_error * delay_chain

    # Function to update proactive control signals
    def updateProactiveControlSignals(self, stimuli, activity, prediction, negative_valence):
        """
        Update the proactive control weights based on the stimuli, activity, prediction, and negative valence.

        Args:
            stimuli (numpy.ndarray): The current stimuli vector.
            activity (numpy.ndarray): The activity vector of the response units.
            prediction (numpy.ndarray): The predicted value vector.
            negative_valence (float): The negative valence associated with a bad outcome.
        """
        inhibition_signal = self.inhibition(activity, prediction)
        if inhibition_signal > self.inhibition_threshold:
            self.w_proactiveControl += self.learning_rate * inhibition_signal * stimuli * negative_valence

    # Function to update reactive control signals
    def updateReactiveControlSignals(self, valence_signals, action_was_executed):
        """
        Update the reactive control weights based on the valence of the outcome and whether an action was executed.

        Args:
            valence_signals (numpy.ndarray): The valence signals associated with each response.
            action_was_executed (bool): Whether an action was executed in the current trial.
        """
        self.learnReactiveControlWeights(valence_signals, action_was_executed)

    # Function to simulate one step of the model
    def step(self, stimuli, actual_roc, reward_vec, action_was_executed=False):
        """
        Simulate one step of the PRO-control model.

        Args:
            stimuli (numpy.ndarray): The current stimuli vector.
            actual_roc (numpy.ndarray): The actual response-outcome conjunction vector.
            reward_vec (numpy.ndarray): The reward vector for the current trial.
            action_was_executed (bool): Whether an action was executed in the previous trial.

        Returns:
            numpy.ndarray: The response vector.
            numpy.ndarray: The positive surprise signal.
            numpy.ndarray: The negative surprise signal.
            bool: Whether an action was executed in the current trial.
        """
        # 1. Update the delay chain
        self.delay_chain = np.roll(self.delay_chain, -1, axis=1)
        self.delay_chain[:, 0] = stimuli

        # 2. Predict response-outcome conjunctions
        predicted_roc = self.responseOutcomeConj(stimuli)

        # 3. Predict values
        predicted_values = self.predictValues(self.delay_chain)

        # 4. Calculate the unit response
        response = self.unitResponse(stimuli)

        # 5. Calculate surprise
        wP, wN = self.calculateSurprise(actual_roc, predicted_roc)

        # 6. Update value predictions
        self.updateValuePredictions(reward_vec, predicted_values, self.delay_chain)

        # 7. Update proactive control signals
        self.updateProactiveControlSignals(stimuli, response, predicted_values, self.negative_valence)

        # 8. Update reactive control signals
        self.updateReactiveControlSignals(self.negativeSurprize, action_was_executed)

        # 9. Update previous response
        self.previous_response = response

        # 10. Determine if an action was executed
        if np.max(response) > self.action_threshold:
            action_was_executed = True

        # 11. Learn response-outcome weights (Controller)
        self.learnResponseOutcomeWeights(actual_roc, predicted_roc, stimuli, action_was_executed)

        return response, wP, wN, action_was_executed

# Example Usage
model = PROControlModel(stimuli_size=100, num_responses=4)

# Simulate a sequence of trials
for _ in range(10):
    stimuli = np.random.rand(model.stimuli_size)  # Example stimuli
    actual_roc = np.random.rand(model.num_responses)  # Example actual response-outcome conjunction
    reward_vec = np.random.rand(model.num_responses)  # Example reward vector
    action_was_executed = False  # Initialize action_was_executed flag

    response, wP, wN, action_was_executed = model.step(stimuli, actual_roc, reward_vec, action_was_executed)

    # ... continue with other model steps