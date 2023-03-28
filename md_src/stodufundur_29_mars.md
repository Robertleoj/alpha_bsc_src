



# What need

- AlphaZero review

- Describe how we measure performance of algorithm with alpha-beta
    - How policy error is measured
    - How value error is measured

- Default run
    - plot progress, and report errors
    - report total playouts performed
    - plot performance as a function of total playouts

- Mention credit assignment problem

- Motivation for endgame playouts
    - In the beginning of training, target evaluation is useless
    - Because network cannot evaluate, policy target is also useless


- Describe how we analyze training data eval target quality

- Show the evaluations for the default run, showing that target is better towards the end of the game
    - Also show how unstable the quality is in the default run

- Describe endgame playouts, and how it capitalizes on this

- Show how it performs
    - Show performance as a function
        - generations
        - total playouts
    - Show how the quality of the training data improves towards the start of the game.
    - Show how steadily the performance improves vs the default run.

- Do in depth analysis of the zig-zag-pattern. Why does it happen?
    - Check out the evaluation of an average position. If it is much more likely to be a win, then it makes sense.

- Plan going forward
    - Run experiemts longer
    - analyze data more
    - Write paper
    - get $$

- Change is technologies
    - Change database to Sqlite3
    - Move training data to BSON
    - No Azure cloud, just local machine

- Risk analysis

    - Bug in the code preventing our results from being valid
        - Fix the bug and rerun experiments 
        - ~ 2 days.

    - What if the servers go down?
        - We can always run on our local machines
        - They are as powerful as the servers.
        - Close to no lost time.

    - Loss of data.
        - Run experiments again - takes 2 days
        - Unlikely to happen
          - We have backups on Github and multiple local machines
        - ~ 2 days

    - What if all our results are not ready in time?
        - There is nothing to do about that, really.
        - We can show off our current results.
            - results for endgame playouts already seem significant.
            - imporovements in Connect4.


- Working hour analysis `>:(`
    - done

