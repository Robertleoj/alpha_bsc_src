create table games (
    id int auto_increment,
    game_name varchar(45) unique not null,
    primary key (id)
);

insert into games (id, game_name) 
values
(1, "connect4"),
(2, "breakthrough");

create table generations (
    id int auto_increment,
    game_id int not null,
    generation_num int not null,
    created_at timestamp default current_timestamp,
    primary key (id),
    unique (game_id, generation_num),
    foreign key (game_id) references games (id)
);

insert into generations (game_id, generation_num)
values
(1, 0),
(2, 0);

create table training_data(
    id int auto_increment,
    generation_id int not null,
    state blob not null,
    policy blob not null,
    outcome float,
    created_at timestamp default current_timestamp,
    primary key (id),
    foreign key (generation_id) references generations (id)
);

