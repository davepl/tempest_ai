import time
import traceback
import torch
import multiprocessing as mp
from queue import Empty
import setproctitle # Import the library

# Assuming config and aimodel are accessible (adjust paths/imports if needed)
# If run as a separate process, ensure Python path allows finding these
try:
    from config import RL_CONFIG, LATEST_MODEL_PATH, DEVICE
    from aimodel import DQNAgent # DQNAgent needs to be importable
except ImportError as e:
    print(f"[TrainingWorker] Import Error: {e}. Ensure training.py is run where config/aimodel are importable.")
    # Depending on structure, might need relative imports or sys.path modification before imports
    import sys, os
    # Simple attempt assuming it's run from the parent directory
    sys.path.append(os.path.dirname(__file__))
    try:
        from config import RL_CONFIG, LATEST_MODEL_PATH, DEVICE
        from aimodel import DQNAgent
    except ImportError:
         raise # Re-raise if still failing

def training_worker(train_batch_queue: mp.Queue, loss_queue: mp.Queue, shutdown_event: mp.Event, state_size: int, action_size: int):
    """
    Worker process dedicated to training the DQN agent.
    Receives batches of experiences and sends back loss values.
    """
    # Limit PyTorch threads for this process
    try:
         torch.set_num_threads(1)
         # Optionally, disable dynamic parallelism if needed
         # torch.set_num_interop_threads(1)
         print(f"[TrainingWorker] Set torch num_threads=1")
    except Exception as thread_err:
         print(f"[TrainingWorker] Warning: Failed to set torch threads: {thread_err}")
         
    setproctitle.setproctitle("python_training") # Set the process title
    print("[TrainingWorker] Starting...")
    try:
        # State and action sizes are now passed as arguments
        if state_size is None or action_size is None:
             print("[TrainingWorker] Error: State or action size not provided.")
             return # Cannot proceed

        # Initialize the agent *within this process*
        # Make sure DQNAgent init doesn't start background threads anymore
        agent = DQNAgent(state_size, action_size)
        print(f"[TrainingWorker] DQNAgent initialized on device: {DEVICE}")

        # Load existing model if available
        try:
            if LATEST_MODEL_PATH.exists():
                # Load now returns (success, metrics_state)
                success, _ = agent.load(LATEST_MODEL_PATH) # Ignore metrics state
                if success:
                     print(f"[TrainingWorker] Loaded model weights from {LATEST_MODEL_PATH}")
                else:
                     print(f"[TrainingWorker] Failed to load model weights from {LATEST_MODEL_PATH}. Starting fresh.")
            else:
                 print(f"[TrainingWorker] No model found at {LATEST_MODEL_PATH}, starting fresh.")
        except Exception as load_err:
            print(f"[TrainingWorker] Error loading model: {load_err}. Starting fresh.")
            traceback.print_exc()

        last_save_time = time.time()
        batches_processed = 0

        while not shutdown_event.is_set():
            try:
                # Get a batch from the main process (blocking with timeout)
                batch = train_batch_queue.get(timeout=1.0)

                if batch is None: # Check for potential poison pill
                    print("[TrainingWorker] Received None batch, potentially stopping.")
                    continue

                # Perform a training step
                # DQNAgent.train_step should be modified to accept a batch
                loss = agent.train_step(batch)

                if loss is not None:
                     batches_processed += 1
                     # Send loss back to the main process
                     try:
                         # Convert tensor loss to float before sending
                         loss_queue.put(loss.item() if isinstance(loss, torch.Tensor) else float(loss), block=False)
                     except mp.queues.Full:
                         # If the main process isn't consuming losses fast enough,
                         # we might skip sending some to avoid blocking the trainer.
                         # print("[TrainingWorker] Warning: Loss queue full, discarding loss.")
                         pass

                # Periodically save the model
                current_time = time.time()
                if current_time - last_save_time >= RL_CONFIG.save_interval_seconds:
                     if batches_processed > 0: # Only save if training happened
                          print(f"[TrainingWorker] Saving model after {batches_processed} batches...")
                          agent.save(LATEST_MODEL_PATH)
                          last_save_time = current_time
                          batches_processed = 0 # Reset counter after save
                     else:
                          # If no batches were processed, maybe reload in case inference process saved newer
                          try:
                              agent.load(LATEST_MODEL_PATH)
                              # print(f"[TrainingWorker] Reloaded model weights from {LATEST_MODEL_PATH} during idle save interval.")
                          except Exception:
                              # Ignore load errors here, keep existing weights
                              pass
                          last_save_time = current_time # Still update time to avoid rapid checks


            except Empty:
                # Queue was empty, just loop again
                # Optional: Check if model file updated by inference worker and reload
                # agent.load(LATEST_MODEL_PATH) # Be careful about load frequency/errors
                continue
            except (KeyboardInterrupt, SystemExit):
                print("[TrainingWorker] Interrupted, shutting down...")
                break
            except Exception as e:
                print(f"[TrainingWorker] Error in training loop: {e}")
                traceback.print_exc()
                # Avoid tight loop on persistent error
                time.sleep(0.5)

        print("[TrainingWorker] Shutdown event set or loop exited.")
        # Final save before exiting
        if batches_processed > 0: # Only save if training happened since last save
             print("[TrainingWorker] Performing final model save...")
             agent.save(LATEST_MODEL_PATH)
        print("[TrainingWorker] Exiting.")

    except Exception as init_err:
         print(f"[TrainingWorker] Fatal error during initialization: {init_err}")
         traceback.print_exc()
