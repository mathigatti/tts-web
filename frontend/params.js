function ptq(q)
{
/* parse the query */
var x = q.replace(/;/g, '&').split('&'), i, name, t;
/* q changes from string version of query to object */
for (q={}, i=0; i<x.length; i++)
{
t = x[i].split('=', 2);
name = t[0];
if (!q[name])
q[name] = [];
if (t.length > 1)
{
q[name][q[name].length] = t[1];
}
/* next two lines are nonstandard */
else
q[name][q[name].length] = true;
}
return q;
}
function param() {
return ptq(location.search.substring(1));
}
