# main.py
from game import Game
from players import QLearningAgent

def run_training_session():
    game = Game(player_count=6)
    # Initialize 6 AI agents
    game.players = [QLearningAgent(f"AI_{i}", "Cairo") for i in range(6)]
    
    for episode in range(1000): # Run 1000 games
        state = game.reset()
        done = False
        while not done:
            for player in game.players:
                # 1. Player sees the board
                obs = game.get_observation(player)
                # 2. Player chooses action
                action = player.get_action(obs)
                # 3. Game executes action
                next_obs, reward, done = game.step(player, action)
                # 4. Player learns from result
                player.learn(obs, action, reward, next_obs)
                
                if done: break

if __name__ == "__main__":
    run_training_session()