create table games (
    id integer primary key asc,
    game_name varchar(45) unique not null
);

insert into games (game_name) 
values
("connect4"),
("breakthrough");

create table generations (
    id integer primary key asc,
    game_id integer not null,
    generation_num integer not null,
    created_at timestamp default current_timestamp,
    unique (game_id, generation_num),
    foreign key (game_id) references games (id)
);

insert into generations (game_id, generation_num)
values
((select id from games where game_name = 'connect4'), 0),
((select id from games where game_name = 'breakthrough'), 0);

create table training_data(
    id integer primary key asc,
    generation_id integer not null,
    state blob not null,
    policy blob not null,
    outcome float,
    player tinyint not null,
    moves text,
    created_at timestamp default current_timestamp,
    foreign key (generation_id) references generations (id)
);

create table losses(
    id integer primary key asc,
    generation_id integer not null,
    iteration integer not null,
    loss float not null,
    created_at timestamp default current_timestamp,
    foreign key (generation_id) references generations (id)
);



