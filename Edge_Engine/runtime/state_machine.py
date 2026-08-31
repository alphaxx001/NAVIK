class NavigationStateMachine:
    def __init__(self):
        self.state = "GNSS"
        self.history = []
        
    def transition(self, new_state, reason=""):
        if new_state != self.state:
            # print(f"[StateMachine] Transition: {self.state} -> {new_state} | Reason: {reason}")
            self.history.append((self.state, new_state, reason))
            self.state = new_state
            
    def update(self, gnss_available, has_map_candidates):
        if gnss_available:
            self.transition("GNSS", "GNSS signal acquired")
        else:
            if has_map_candidates:
                self.transition("MAP_CONTEXT", "GNSS lost, valid map candidates available")
            else:
                self.transition("DR_FALLBACK", "No valid map candidates found within radius")
                
    def get_state(self):
        return self.state
