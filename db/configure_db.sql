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


