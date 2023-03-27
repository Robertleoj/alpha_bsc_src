create table generations (
    id integer primary key asc,
    generation_num integer not null,
    created_at timestamp default current_timestamp,
    unique (generation_num)
);

-- When fixing the db comment the following 2 lines out.
-- insert into generations (generation_num)
-- values (0);

create table losses(
    id integer primary key asc,
    generation_id integer not null,
    iteration integer not null,
    loss float not null,
    created_at timestamp default current_timestamp,
    foreign key (generation_id) references generations (id)
);

create table ground_truth_evals(
    id integer primary key asc,
    generation_id integer not null,
    moves text not null,
    search_depth integer not null,

    policy_target json not null,
    value_target float not null,

    policy_prior json not null,
    policy_mcts json not null,

    nn_value float not null,
    nn_value_error float not null,

    mcts_value float not null,
    mcts_value_error float not null,

    prior_error float not null,
    mcts_error float not null,

    created_at timestamp default current_timestamp,
    foreign key (generation_id) references generations (id)
);




