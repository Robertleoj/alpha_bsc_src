create table generations (
    id int auto_increment,
    generation_num int unique not null,
    created_at timestamp default current_timestamp,
    primary key (id)
);

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
