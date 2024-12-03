    def forward(self, s):
        rs = [] # Store r values for each layer (ONE-HOT)
        ps = [] # Store predictions for each layer
        modulation = None # Initialize modulation

        for i in range(self.n_layers - 1, -1, -1):
            r_prev_index = self.WM[i]  # Get current WM content (index)
            r_prev = F.one_hot(r_prev_index.long(), num_classes=self.n_representations).float()  # One-hot encode
            r = self.WM_gates[i](s, r_prev, i) # Pass one-hot r_prev

            # Update WM (store index of the chosen stimulus)
            self.WM[i] = r.argmax() # Store index of active WM unit

            rs.append(r) # Store r (ONE-HOT) <--- This is the crucial change

            self.eligibility_traces[i] = self.lambdas[i] * self.eligibility_traces[i] + s

            wm_mask = F.one_hot(self.WM[i].long(), num_classes=self.n_representations).float()
            p = self.layers[i](r, wm_mask, modulation) # Pass one-hot r
            ps.append(p)

            if i > 0:
                modulation = p.flatten()

        response_logits = torch.matmul(ps[-1], self.W_response) # Use predictions from lowest layer
        response_probs = F.softmax(response_logits * self.gamma, dim=-1) # Apply softmax
        return response_probs, rs, ps # Return response probabilities