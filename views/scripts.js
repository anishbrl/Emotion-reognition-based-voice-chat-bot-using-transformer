const jokeContainer = document.getElementById("joke");
const btn = document.getElementById("btn");
const url = "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist,explicit&type=single";
const emoji = document.querySelector("span");

canclick = true;

let getJoke = () => {
    if (canclick) {
        canclick = false;
setTimeout(() => {
    jokeContainer.classList.remove("fade");
    btn.classList.remove("clicked");
    emoji.classList.remove("rotate");
    fetch(url)
    .then(data => data.json())
    .then(item =>{
        jokeContainer.textContent = `${item.joke}`;
        jokeContainer.classList.add("fade");
    });
    canclick = true;
}, 1000);
}};
btn.addEventListener("click",()=>{
    btn.classList.add("clicked");
    emoji.classList.add("rotate");
    getJoke();
});
getJoke();